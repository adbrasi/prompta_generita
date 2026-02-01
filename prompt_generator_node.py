import random
import re
import os
import json
import logging
import time
import urllib.request
import urllib.error
import urllib.parse
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OutfitDistribution:
    """Represents an outfit and its distribution weight"""
    outfit: str
    weight: float = 1.0

class PromptFileCache:
    """Caches file contents to avoid repeated I/O operations
    
    Note: This cache only stores file contents, NOT random selections.
    Each prompt generation creates fresh random selections from cached file data.
    """
    
    def __init__(self):
        self._cache: Dict[str, List[str]] = {}
        self._file_mtimes: Dict[str, float] = {}
    
    def get_lines(self, filepath: str) -> List[str]:
        """Get lines from file with caching and modification time checking"""
        path = Path(filepath)
        
        if not path.exists():
            logger.warning(f"File not found: {filepath}")
            return []
        
        current_mtime = path.stat().st_mtime
        
        # Check if file was modified or not in cache
        if (filepath not in self._cache or 
            filepath not in self._file_mtimes or 
            current_mtime != self._file_mtimes[filepath]):
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                self._cache[filepath] = lines
                self._file_mtimes[filepath] = current_mtime
                logger.info(f"Loaded {len(lines)} lines from {filepath}")
            except Exception as e:
                logger.error(f"Error reading file {filepath}: {e}")
                return []
        
        return self._cache[filepath]

class OutfitParser:
    """Handles parsing and distribution of outfits using /CUT syntax"""
    
    SEPARATOR_PATTERN = re.compile(r'/cut', re.IGNORECASE)  # Case-insensitive pattern
    
    @classmethod
    def parse_outfits(cls, outfit_string: str) -> List[OutfitDistribution]:
        """Parse outfit string with /CUT separators into distributions (case-insensitive)"""
        if not outfit_string.strip():
            return [OutfitDistribution("")]
        
        # Split by /CUT (case-insensitive) and clean up
        parts = cls.SEPARATOR_PATTERN.split(outfit_string)
        parts = [part.strip() for part in parts if part.strip()]  # Remove empty parts
        
        if not parts:
            return [OutfitDistribution("")]
        
        # Create equal weight distributions
        return [OutfitDistribution(outfit) for outfit in parts]
    
    @classmethod
    def distribute_outfits(cls, outfits: List[OutfitDistribution], count: int, rng: random.Random) -> List[str]:
        """Distribute outfits across count items based on weights"""
        if not outfits or count <= 0:
            return []
        
        # Calculate how many items each outfit should get
        total_weight = sum(outfit.weight for outfit in outfits)
        distributions = []
        
        remaining_count = count
        for i, outfit in enumerate(outfits):
            if i == len(outfits) - 1:  # Last outfit gets remaining
                outfit_count = remaining_count
            else:
                outfit_count = int((outfit.weight / total_weight) * count)
                remaining_count -= outfit_count
            
            distributions.extend([outfit.outfit] * outfit_count)
        
        # Shuffle to randomize distribution
        rng.shuffle(distributions)
        return distributions

class PromptCleaner:
    """Handles cleaning and formatting of generated prompts"""
    
    @staticmethod
    def cleanup_tags(text: str) -> str:
        """Clean up comma formatting and remove empty tags"""
        if not text:
            return ""
        
        # Replace multiple commas with single comma
        text = re.sub(r'\s*,\s*(?:,\s*)+', ', ', text)
        
        # Remove leading/trailing commas and spaces
        text = re.sub(r'^[,\s]+|[,\s]+$', '', text)
        
        # Split, clean, and rejoin tags
        tags = []
        for tag in text.split(','):
            tag = tag.strip()
            if tag and tag not in tags:  # Remove duplicates and empty tags
                tags.append(tag)
        
        return ', '.join(tags)
    
    @staticmethod
    def combine_outfit_and_prompt(outfit: str, prompt: str) -> str:
        """Combine outfit and prompt with proper formatting"""
        parts = []
        if outfit.strip():
            parts.append(outfit.strip())
        if prompt.strip():
            parts.append(prompt.strip())
        return ', '.join(parts)

class CivitaiAPI:
    BASE_URL = "https://civitai.com/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Packreator/1.0",
        }
        if api_key:
            self.base_headers["Authorization"] = f"Bearer {api_key}"

    @staticmethod
    def _redact_token(url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        if "token" in query:
            query["token"] = ["***"]
        new_query = urllib.parse.urlencode(query, doseq=True)
        return urllib.parse.urlunparse(parsed._replace(query=new_query))

    def _build_url(self, endpoint: str, include_token: bool = False) -> str:
        base_url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        if not include_token or not self.api_key:
            return base_url
        parsed = urllib.parse.urlparse(base_url)
        query = urllib.parse.parse_qs(parsed.query)
        query.setdefault("token", [self.api_key])
        new_query = urllib.parse.urlencode(query, doseq=True)
        return urllib.parse.urlunparse(parsed._replace(query=new_query))

    def _request(self, endpoint: str) -> Optional[Dict[str, Any]]:
        url = self._build_url(endpoint, include_token=False)
        req = urllib.request.Request(url, headers=self.base_headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 204:
                    return None
                body = resp.read().decode("utf-8")
            return json.loads(body) if body else None
        except urllib.error.HTTPError as http_err:
            if http_err.code == 403 and self.api_key:
                retry_url = self._build_url(endpoint, include_token=True)
                try:
                    retry_req = urllib.request.Request(retry_url, headers=self.base_headers, method="GET")
                    with urllib.request.urlopen(retry_req, timeout=30) as resp:
                        if resp.status == 204:
                            return None
                        body = resp.read().decode("utf-8")
                    return json.loads(body) if body else None
                except urllib.error.HTTPError as retry_err:
                    safe_url = self._redact_token(retry_url)
                    logger.warning(f"Erro de API do Civitai (retry token): {retry_err} (URL: {safe_url})")
                    return {"error": str(retry_err), "status_code": retry_err.code, "hint": "Falha mesmo com token na query string."}
            safe_url = self._redact_token(url)
            logger.warning(f"Erro de API do Civitai: {http_err} (URL: {safe_url})")
            hint = ""
            if http_err.code == 403:
                hint = "403 Forbidden. Verifique se a civitai_api_key é necessária ou tem acesso ao modelo."
            return {"error": str(http_err), "status_code": http_err.code, "hint": hint}
        except urllib.error.URLError as req_err:
            logger.warning(f"Erro de requisição do Civitai: {req_err}")
            return {"error": str(req_err)}
        except Exception as e:
            logger.warning(f"Erro inesperado no Civitai: {e}")
            return {"error": str(e)}

    def get_model_info(self, model_id: int) -> Optional[Dict[str, Any]]:
        return self._request(f"/models/{model_id}")

    def get_model_version_info(self, version_id: int) -> Optional[Dict[str, Any]]:
        return self._request(f"/model-versions/{version_id}")

def parse_civitai_input(url_or_id: str) -> Tuple[Optional[int], Optional[int]]:
    url_or_id = str(url_or_id).strip()
    if not url_or_id:
        return None, None

    air_match = re.search(r'(?:urn:air:sdxl:lora:)?(?:civitai:)?(\d+)@(\d+)', url_or_id)
    if air_match:
        model_id_str, version_id_str = air_match.groups()
        return int(model_id_str), int(version_id_str)

    if url_or_id.isdigit():
        return int(url_or_id), None

    try:
        parsed_url = urllib.parse.urlparse(url_or_id)
        path_parts = [p for p in parsed_url.path.split('/') if p]
        query_params = urllib.parse.parse_qs(parsed_url.query)

        model_id: Optional[int] = None
        version_id: Optional[int] = None

        if 'modelVersionId' in query_params:
            version_id = int(query_params['modelVersionId'][0])

        if "models" in path_parts:
            model_index = path_parts.index("models")
            if model_index + 1 < len(path_parts) and path_parts[model_index + 1].isdigit():
                model_id = int(path_parts[model_index + 1])

        if not version_id and "model-versions" in path_parts:
            version_index = path_parts.index("model-versions")
            if version_index + 1 < len(path_parts) and path_parts[version_index + 1].isdigit():
                version_id = int(path_parts[version_index + 1])

        return model_id, version_id
    except Exception:
        return None, None

def get_civitai_details(link_or_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    model_id, version_id = parse_civitai_input(link_or_id)

    if not model_id and not version_id:
        return {"error": "Não foi possível extrair um ID de modelo ou versão válido da entrada."}

    api = CivitaiAPI(api_key=api_key)

    try:
        if not model_id and version_id:
            temp_version_info = api.get_model_version_info(version_id)
            if temp_version_info and 'modelId' in temp_version_info:
                model_id = temp_version_info['modelId']
            else:
                return {"error": f"Não foi possível encontrar o modelo para a versão ID {version_id}."}

        model_info = api.get_model_info(model_id)
        if not model_info or "error" in model_info:
            if isinstance(model_info, dict) and model_info.get("status_code") == 403:
                if not api_key:
                    return {"error": "Civitai retornou 403. Informe uma civitai_api_key válida para acessar o modelo."}
                return {"error": "Civitai retornou 403. Verifique se sua civitai_api_key tem acesso ao modelo."}
            return {"error": f"Falha ao buscar informações do modelo ID {model_id}."}

        if not version_id:
            if model_info.get("modelVersions"):
                version_id = model_info["modelVersions"][0].get('id')
                if not version_id:
                    return {"error": "Não foi possível encontrar o ID na versão mais recente."}
            else:
                return {"error": "O modelo não tem nenhuma versão listada."}

        version_info = api.get_model_version_info(version_id)
        if not version_info or "error" in version_info:
            if isinstance(version_info, dict) and version_info.get("status_code") == 403:
                if not api_key:
                    return {"error": "Civitai retornou 403 ao buscar versão. Informe uma civitai_api_key válida."}
                return {"error": "Civitai retornou 403 ao buscar versão. Verifique se sua civitai_api_key tem acesso."}
            return {"error": f"Falha ao buscar informações da versão ID {version_id}."}

    except Exception as e:
        return {"error": f"Erro ao processar: {str(e)}"}

    model_name = model_info.get('name', 'N/A')
    version_name = version_info.get('name', 'N/A')

    description_html = model_info.get('description')
    description_text = re.sub('<[^<]+?>', '', description_html).strip() if description_html else "Nenhuma descrição fornecida."
    trained_words = version_info.get('trainedWords', [])

    return {
        "model_name": model_name,
        "version_name": version_name,
        "model_id": model_id,
        "version_id": version_id,
        "air_id": f"{model_id}@{version_id}",
        "description": description_text,
        "trained_words": trained_words
    }

class EnhancedRandomGenerator:
    """Improved random number generator with better seeding and true randomness"""
    
    def __init__(self, seed: Optional[int] = None):
        # Always create a new random instance to avoid state pollution
        if seed is None or seed < 0:
            # Use time-based seed for true randomness on each call
            actual_seed = int(time.time() * 1000000) % (2**32)
            self.rng = random.Random(actual_seed)
            self.seed = None
            logger.debug(f"Using time-based seed: {actual_seed}")
        else:
            self.rng = random.Random(seed)
            self.seed = seed
            logger.debug(f"Using provided seed: {seed}")
    
    def sample_unique(self, population: List, k: int) -> List:
        """Sample without replacement, handling edge cases"""
        if not population:
            return []
        
        k = min(k, len(population))
        if k <= 0:
            return []
        
        # Create a copy to avoid modifying original
        pop_copy = list(population)
        return self.rng.sample(pop_copy, k)
    
    def shuffle(self, x: List) -> None:
        """Shuffle list in place"""
        self.rng.shuffle(x)

class CharacterSheetCache:
    """Cache para planilhas de personagens com recarga automática por mtime."""

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._file_mtimes: Dict[str, float] = {}

    def load(self, filepath: str) -> pd.DataFrame:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Planilha não encontrada: {filepath}")

        current_mtime = path.stat().st_mtime
        if filepath not in self._cache or self._file_mtimes.get(filepath) != current_mtime:
            df = pd.read_excel(filepath)
            df.columns = [str(col).strip() for col in df.columns]
            self._cache[filepath] = df
            self._file_mtimes[filepath] = current_mtime
            logger.info(f"Planilha carregada: {filepath} ({len(df)} linhas)")

        return self._cache[filepath]

class PackPromptGeneratorCore:
    """Gerador que integra planilha + seleção de outfits + prompts de seção."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.sections_path = self.base_path / "sections"
        self.file_cache = PromptFileCache()
        self.sheet_cache = CharacterSheetCache()
        self._sequential_state_path = self.base_path / "sequential_state.json"

    _SEQUENTIAL_RANGE_PATTERN = re.compile(r"^\s*(\d+)\s*~\s*(\d+)\s*$")

    @staticmethod
    def _normalize_key(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    @staticmethod
    def _unique_preserve(items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            key = PackPromptGeneratorCore._normalize_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        normalized_map = {
            re.sub(r"[_\s]+", "", str(col).strip().lower()): col for col in df.columns
        }
        for cand in candidates:
            key = re.sub(r"[_\s]+", "", cand.strip().lower())
            if key in normalized_map:
                return normalized_map[key]
        return None

    @staticmethod
    def _is_empty_civitai_id(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (int, float)) and value == 0:
            return True
        text = str(value).strip()
        if not text:
            return True
        if text.lower() in {"none", "null", "nan", "0", "false"}:
            return True
        return False

    @staticmethod
    def _process_civitai_id(civitai_id: Any, prefix: str = "urn:air:sdxl:lora:civitai:") -> str:
        if civitai_id is None or (isinstance(civitai_id, float) and pd.isna(civitai_id)):
            return ""
        civitai_str = str(civitai_id).strip()
        if not civitai_str or civitai_str.lower() == "nan":
            return ""
        if prefix and civitai_str.startswith(prefix):
            return civitai_str[len(prefix):]
        if "civitai:" in civitai_str:
            return civitai_str.split("civitai:", 1)[1]
        if "civitai/" in civitai_str:
            return civitai_str.split("civitai/", 1)[1]
        if civitai_str.startswith("urn:") and ":" in civitai_str:
            return civitai_str.split(":")[-1]
        return civitai_str

    def _parse_sequential_range(self, value: Any) -> Optional[Tuple[int, int]]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        match = self._SEQUENTIAL_RANGE_PATTERN.match(text)
        if not match:
            raise ValueError("Faixa sequencial inválida. Use o formato 5~10.")
        start = int(match.group(1))
        end = int(match.group(2))
        if start <= 0 or end <= 0:
            raise ValueError("Faixa sequencial inválida. Use números >= 1.")
        if start > end:
            raise ValueError("Faixa sequencial inválida. O início não pode ser maior que o fim.")
        return start, end

    @staticmethod
    def _build_sequential_state_key(spreadsheet: str,
                                    character_filter: str,
                                    random_pool: str,
                                    seq_range: Tuple[int, int]) -> str:
        start, end = seq_range
        filter_part = (character_filter or "").strip().lower()
        pool_part = (random_pool or "").strip().lower()
        return f"{spreadsheet}|filter={filter_part}|pool={pool_part}|range={start}~{end}"

    def _load_sequential_state(self) -> Dict[str, int]:
        if not self._sequential_state_path.exists():
            return {}
        try:
            with open(self._sequential_state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Falha ao ler estado sequencial: %s", e)
            return {}
        if not isinstance(data, dict):
            logger.warning("Estado sequencial inválido (esperado dict). Resetando.")
            return {}
        cleaned: Dict[str, int] = {}
        for key, value in data.items():
            try:
                cleaned[key] = int(value)
            except (TypeError, ValueError):
                continue
        return cleaned

    def _save_sequential_state(self, state: Dict[str, int]) -> None:
        try:
            with open(self._sequential_state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=True, indent=2, sort_keys=True)
        except Exception as e:
            logger.warning("Falha ao salvar estado sequencial: %s", e)

    def _load_outfits(self, row: pd.Series, outfit_columns: List[str]) -> List[str]:
        outfits = []
        for col in outfit_columns:
            if col in row.index and not pd.isna(row[col]):
                value = str(row[col]).strip()
                if value:
                    outfits.append(value)
        return self._unique_preserve(outfits)

    def _select_outfits_no_repeat(self, outfits: List[str], count: int, rng: EnhancedRandomGenerator) -> List[str]:
        if count <= 0 or not outfits:
            return []
        count = min(count, len(outfits))
        return rng.sample_unique(outfits, count)

    def _allocate_outfits(self,
                          outfits: List[str],
                          counts: Dict[str, int],
                          rng: EnhancedRandomGenerator,
                          preferred: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
        available = self._unique_preserve(outfits)
        available_map = {self._normalize_key(o): o for o in available}
        used = set()
        selections: Dict[str, List[str]] = {}

        def take_preferred(section: str, count: int) -> List[str]:
            chosen = []
            if not preferred:
                return chosen
            for item in preferred.get(section, []):
                key = self._normalize_key(item)
                if key in available_map and key not in used:
                    chosen.append(available_map[key])
                    used.add(key)
                if len(chosen) >= count:
                    break
            return chosen

        def fill_random(count: int) -> List[str]:
            remaining = [o for o in available if self._normalize_key(o) not in used]
            chosen = []
            if not remaining:
                return chosen
            take = min(count, len(remaining))
            chosen.extend(rng.sample_unique(remaining, take))
            for item in chosen:
                used.add(self._normalize_key(item))
            return chosen

        for section in ("s1", "s2", "s3"):
            count = max(0, counts.get(section, 0))
            if count == 0:
                selections[section] = []
                continue
            chosen = take_preferred(section, count)
            if len(chosen) < count:
                chosen.extend(fill_random(count - len(chosen)))
            selections[section] = chosen

        return selections

    def _select_section_prompts(self,
                                filename: str,
                                count: int,
                                pingpong: bool,
                                rng: EnhancedRandomGenerator) -> List[str]:
        if count <= 0:
            return []
        path = self.sections_path / filename
        lines = self.file_cache.get_lines(str(path))
        if not lines:
            return []

        if not pingpong:
            return rng.sample_unique(lines, min(count, len(lines)))

        # Pingpong: alterna entre arquivo principal e seu par V2 (se existir)
        counterpart = None
        if filename.lower().endswith("v2.txt"):
            counterpart = filename[:-6] + ".txt"
        else:
            counterpart = filename[:-4] + "V2.txt"
        counterpart_path = self.sections_path / counterpart
        lines_b = self.file_cache.get_lines(str(counterpart_path)) if counterpart_path.exists() else []
        if not lines_b:
            return rng.sample_unique(lines, min(count, len(lines)))

        chosen = []
        for i in range(count):
            source = lines if i % 2 == 0 else lines_b
            if not source:
                source = lines_b if source is lines else lines
            if source:
                chosen.append(rng.rng.choice(source))
        return chosen

    @staticmethod
    def _build_prompt(character_tags: str, outfit: str, section_prompt: str) -> str:
        parts = []
        if character_tags.strip():
            parts.append(character_tags.strip())
        if outfit.strip():
            parts.append(outfit.strip())
        if section_prompt.strip():
            parts.append(section_prompt.strip())
        combined = ", ".join(parts)
        return PromptCleaner.cleanup_tags(combined)

    @staticmethod
    def _parse_backgrounds(text: str) -> List[str]:
        if not text:
            return []
        raw = text.strip()
        if not raw:
            return []
        parts = re.split(r"/cut|/|\\n", raw, flags=re.IGNORECASE)
        cleaned = [part.strip() for part in parts if part.strip()]
        return PackPromptGeneratorCore._unique_preserve(cleaned)

    @staticmethod
    def _distribute_backgrounds(backgrounds: List[str], total: int) -> List[str]:
        if not backgrounds or total <= 0:
            return []
        if total <= len(backgrounds):
            return backgrounds[:total]
        base = total // len(backgrounds)
        rem = total % len(backgrounds)
        counts = [base] * len(backgrounds)
        if rem:
            counts[-1] += rem
        assignments: List[str] = []
        for bg, cnt in zip(backgrounds, counts):
            if cnt > 0:
                assignments.extend([bg] * cnt)
        return assignments[:total]

    def _call_llm(self,
                  provider: str,
                  api_key: str,
                  model: str,
                  messages: List[Dict[str, str]],
                  response_format: Optional[Dict[str, Any]] = None,
                  timeout: int = 60) -> str:
        if provider == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
        elif provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
        else:
            raise ValueError("Provider inválido. Use 'openrouter' ou 'groq'.")

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }
        if response_format:
            payload["response_format"] = response_format
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        response = json.loads(body)
        return response["choices"][0]["message"]["content"]

    def _extract_json(self, content: str) -> Dict[str, Any]:
        if not content:
            return {}
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Resposta da LLM não é JSON válido. Conteúdo bruto: %s", content)
            return {}

    def _normalize_llm_outfits(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str):
            if re.search(r"/cut", value, flags=re.IGNORECASE):
                items = [part.strip() for part in re.split(r"/cut", value, flags=re.IGNORECASE) if part.strip()]
            else:
                items = [value.strip()] if value.strip() else []
        else:
            items = [str(value).strip()] if str(value).strip() else []
        return self._unique_preserve(items)

    def _clamp_outfits(self, outfits: List[str], count: int) -> List[str]:
        if count <= 0:
            return []
        if len(outfits) <= count:
            return outfits
        return outfits[:count]

    def extrair_info_civitai(self,
                             civitai_context: str,
                             prompt_usuario: str,
                             counts: Dict[str, int],
                             provider: str,
                             api_key: str,
                             model: str) -> Tuple[str, Dict[str, List[str]]]:
        system_prompt = (
            "Você extrai informações de um modelo do Civitai e retorna tags de personagem e outfits. "
            "Retorne SOMENTE JSON válido, com as chaves: "
            "{\"character_tags\": \"...\", \"s1_outfit\": [...], \"s2_outfit\": [...], \"s3_outfit\": [...]} "
            "character_tags deve conter tokens de ativação/trigger words e descrições essenciais do personagem, "
            "sem roupas. Use apenas o que estiver no contexto fornecido."
        )
        user_prompt = (
            f"Comentário do usuário:\n{prompt_usuario or '(vazio)'}\n\n"
            f"Contexto Civitai:\n{civitai_context}\n\n"
            f"Quantidades desejadas:\n"
            f"s1_outfit_count: {counts.get('s1', 0)}\n"
            f"s2_outfit_count: {counts.get('s2', 0)}\n"
            f"s3_outfit_count: {counts.get('s3', 0)}\n"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        content = self._call_llm(
            provider,
            api_key,
            model,
            messages,
            response_format={"type": "json_object"},
        )
        data = self._extract_json(content)
        character_tags = str(data.get("character_tags", "")).strip()
        if not character_tags and data.get("character_description"):
            character_tags = str(data.get("character_description", "")).strip()
        outfits = {
            "s1": self._normalize_llm_outfits(data.get("s1_outfit") or data.get("s1")),
            "s2": self._normalize_llm_outfits(data.get("s2_outfit") or data.get("s2")),
            "s3": self._normalize_llm_outfits(data.get("s3_outfit") or data.get("s3")),
        }
        return character_tags, outfits

    def escolher_outfits_inteligente(self,
                                     outfits_disponiveis: List[str],
                                     prompt_usuario: str,
                                     counts: Dict[str, int],
                                     provider: str,
                                     api_key: str,
                                     model: str,
                                     system_prompt: str,
                                     civitai_context: str = "") -> Dict[str, List[str]]:
        default_system = (
            "Você seleciona outfits de uma lista fornecida para três seções (s1, s2, s3). "
            "Retorne SOMENTE um JSON válido, sem explicações, no formato: "
            "{\"s1_outfit\": [...], \"s2_outfit\": [...], \"s3_outfit\": [...]} "
            "Use apenas outfits da lista. Não invente."
        )
        system_prompt_final = system_prompt.strip() or default_system
        outfits_text = "\n".join(f"- {o}" for o in outfits_disponiveis)
        context_block = f"\n\nContexto Civitai:\n{civitai_context}" if civitai_context else ""
        user_prompt = (
            f"Comentário do usuário:\n{prompt_usuario or '(vazio)'}"
            f"{context_block}\n\n"
            f"Outfits disponíveis:\n{outfits_text}\n\n"
            f"Quantidades:\n"
            f"s1_outfit_count: {counts.get('s1', 0)}\n"
            f"s2_outfit_count: {counts.get('s2', 0)}\n"
            f"s3_outfit_count: {counts.get('s3', 0)}\n"
        )
        messages = [
            {"role": "system", "content": system_prompt_final},
            {"role": "user", "content": user_prompt},
        ]

        content = self._call_llm(
            provider,
            api_key,
            model,
            messages,
            response_format={"type": "json_object"},
        )
        data = self._extract_json(content)
        preferred = {
            "s1": self._normalize_llm_outfits(data.get("s1_outfit")),
            "s2": self._normalize_llm_outfits(data.get("s2_outfit")),
            "s3": self._normalize_llm_outfits(data.get("s3_outfit")),
        }
        return preferred

    def gerar_prompts(self,
                      spreadsheet: str,
                      s1_source: str,
                      s2_source: str,
                      s3_source: str,
                      character_filter: str,
                      sequential_range: str,
                      random_pool: str,
                      civitai_id_input: str,
                      civitai_api_key: str,
                      tags_rule_input: str,
                      s1_prompt_count: int,
                      s2_prompt_count: int,
                      s3_prompt_count: int,
                      s1_outfit_count: int,
                      s2_outfit_count: int,
                      s3_outfit_count: int,
                      background: str,
                      organizacao_inteligente: bool,
                      prompt_usuario: str,
                      system_prompt: str,
                      provider: str,
                      api_key: str,
                      model: str,
                      pingpong: bool,
                      seed: Optional[int] = None) -> Tuple[List[str], str, str, str, str]:
        rng = EnhancedRandomGenerator(seed)

        counts = {
            "s1": max(0, s1_outfit_count),
            "s2": max(0, s2_outfit_count),
            "s3": max(0, s3_outfit_count),
        }

        use_civitai = not self._is_empty_civitai_id(civitai_id_input)

        character_tags = ""
        tags_rule = ""
        pixiv_tag = ""
        civitai_id_output = ""
        selections: Dict[str, List[str]] = {"s1": [], "s2": [], "s3": []}

        if use_civitai:
            if sequential_range and str(sequential_range).strip():
                logger.info("Faixa sequencial ignorada porque o Civitai ID está ativo.")
            if not organizacao_inteligente:
                raise ValueError("Ative organização inteligente para a IA conseguir pegar os dados do Civitai.")
            civitai_id_value = str(civitai_id_input).strip()
            model_id, version_id = parse_civitai_input(civitai_id_value)
            logger.info("Civitai ID informado: %s", civitai_id_value)
            logger.info("Civitai parseado: model_id=%s version_id=%s", model_id, version_id)
            if not model_id and not version_id:
                raise ValueError("Civitai ID inválido. Informe URL, ID numérico ou AIR ID válido.")
            civitai_details = get_civitai_details(civitai_id_value, api_key=civitai_api_key or None)
            if "error" in civitai_details:
                raise ValueError(civitai_details["error"])

            trained_words = civitai_details.get("trained_words") or []
            trained_text = "\n".join(f"- {word}" for word in trained_words) if trained_words else "(sem trained words)"
            civitai_context = (
                f"Modelo: {civitai_details.get('model_name', 'N/A')}\n"
                f"Versão: {civitai_details.get('version_name', 'N/A')}\n"
                f"ID AIR: civitai:{civitai_details.get('air_id', '')}\n"
                f"Descrição:\n{civitai_details.get('description', '')}\n"
                f"Palavras de gatilho:\n{trained_text}"
            )

            if not api_key:
                raise ValueError("API key ausente. Informe a chave do OpenRouter/Groq.")

            character_tags, outfits_llm = self.extrair_info_civitai(
                civitai_context=civitai_context,
                prompt_usuario=prompt_usuario,
                counts=counts,
                provider=provider,
                api_key=api_key,
                model=model,
            )
            character_tags = PromptCleaner.cleanup_tags(character_tags)
            selections = {
                "s1": self._clamp_outfits(outfits_llm.get("s1", []), counts["s1"]),
                "s2": self._clamp_outfits(outfits_llm.get("s2", []), counts["s2"]),
                "s3": self._clamp_outfits(outfits_llm.get("s3", []), counts["s3"]),
            }
            tags_rule = str(tags_rule_input).strip() if tags_rule_input else ""
            civitai_id_output = civitai_details.get("air_id", "") or self._process_civitai_id(civitai_id_value)
        else:
            logger.info("Civitai ID vazio. Usando planilha para seleção de personagem.")
            sheet_path = self.base_path / spreadsheet
            df = self.sheet_cache.load(str(sheet_path))

            tags_rule_col = self._find_column(df, ["TAGS RULE", "tags_rule"])
            civitai_col = self._find_column(df, ["CIVITAI ID", "civitai_id"])
            pixiv_col = self._find_column(df, ["pixiv_tag", "pixivtag"])
            character_col = self._find_column(df, [
                "character_tags", "character_tag", "tags_character", "characterTokens"
            ])
            if not character_col:
                raise ValueError("Coluna de character_tags não encontrada na planilha.")

            outfit_columns = [
                col for col in df.columns if str(col).strip().lower().startswith("outfit_")
            ]
            outfit_columns.sort(key=lambda x: int(re.sub(r"[^0-9]", "", str(x)) or 0))

            if df.empty:
                raise ValueError("Planilha vazia.")

            df_pool = df

            if character_filter and character_filter.strip():
                if not tags_rule_col:
                    raise ValueError("Coluna TAGS RULE não encontrada para filtro.")
                filtro = character_filter.strip().lower()
                mask = df_pool[tags_rule_col].astype(str).str.lower().str.contains(filtro, na=False)
                df_pool = df_pool[mask]
                if df_pool.empty:
                    raise ValueError(f"Nenhum personagem encontrado para filtro: {character_filter}")
            else:
                pool_map = {
                    "all": None,
                    "top 150": 150,
                    "top 100": 100,
                    "top 75": 75,
                    "top 50": 50,
                    "top 25": 25,
                }
                limit = pool_map.get(random_pool, None)
                if limit:
                    df_pool = df_pool.head(limit)
                    if df_pool.empty:
                        raise ValueError("A seleção de top N não contém personagens.")
            seq_range = self._parse_sequential_range(sequential_range)
            if seq_range:
                start, end = seq_range
                if df_pool.empty:
                    raise ValueError("Lista vazia após filtros. Ajuste o filtro ou a faixa sequencial.")
                if end > len(df_pool):
                    raise ValueError(
                        f"Faixa sequencial {start}~{end} excede o total de {len(df_pool)} personagens."
                    )
                state = self._load_sequential_state()
                state_key = self._build_sequential_state_key(
                    spreadsheet=spreadsheet,
                    character_filter=character_filter,
                    random_pool=random_pool,
                    seq_range=seq_range,
                )
                next_index = state.get(state_key, start)
                try:
                    next_index = int(next_index)
                except (TypeError, ValueError):
                    next_index = start
                if next_index < start or next_index > end:
                    next_index = start
                row = df_pool.iloc[next_index - 1]
                following = next_index + 1
                if following > end:
                    following = start
                state[state_key] = following
                self._save_sequential_state(state)
                logger.info(
                    "Faixa sequencial ativa (%s~%s). Selecionado índice %s; próximo %s.",
                    start,
                    end,
                    next_index,
                    following,
                )
            else:
                random_state = rng.rng.randint(0, 2**32 - 1)
                row = df_pool.sample(n=1, random_state=random_state).iloc[0]

            tags_rule = str(row.get(tags_rule_col, "")).strip() if tags_rule_col else ""
            civitai_id_sheet = self._process_civitai_id(row.get(civitai_col, "")) if civitai_col else ""
            pixiv_tag = str(row.get(pixiv_col, "")).strip() if pixiv_col else ""
            character_tags = str(row.get(character_col, "")).strip() if character_col else ""
            if tags_rule_input and str(tags_rule_input).strip():
                tags_rule = str(tags_rule_input).strip()
            civitai_id_output = civitai_id_sheet

            outfits_disponiveis = self._load_outfits(row, outfit_columns)

            preferred = None
            if organizacao_inteligente:
                if not api_key:
                    logger.warning("organização inteligente ativa, mas sem API key. Usando seleção aleatória.")
                else:
                    try:
                        preferred = self.escolher_outfits_inteligente(
                            outfits_disponiveis,
                            prompt_usuario,
                            counts,
                            provider,
                            api_key,
                            model,
                            system_prompt,
                            civitai_context=""
                        )
                    except Exception as e:
                        logger.warning(f"Falha na seleção inteligente: {e}. Usando seleção aleatória.")

            if preferred is None:
                selections = self._allocate_outfits(outfits_disponiveis, counts, rng, preferred=None)
            else:
                selections = self._allocate_outfits(outfits_disponiveis, counts, rng, preferred=preferred)

        prompts = []
        backgrounds = self._parse_backgrounds(background)

        for section, source, prompt_count in [
            ("s1", s1_source, s1_prompt_count),
            ("s2", s2_source, s2_prompt_count),
            ("s3", s3_source, s3_prompt_count),
        ]:
            section_prompts = self._select_section_prompts(source, prompt_count, pingpong, rng)
            if not section_prompts:
                continue
            outfits_section = selections.get(section, [])
            if not outfits_section:
                outfits_assigned = [""] * len(section_prompts)
            else:
                outfits_assigned = OutfitParser.distribute_outfits(
                    [OutfitDistribution(o) for o in outfits_section],
                    len(section_prompts),
                    rng.rng
                )
            for prompt_line, outfit in zip(section_prompts, outfits_assigned):
                combined = self._build_prompt(character_tags, outfit, prompt_line)
                if combined:
                    prompts.append(combined)

        if backgrounds and prompts:
            assignments = self._distribute_backgrounds(backgrounds, len(prompts))
            if assignments:
                updated = []
                for prompt_text, bg in zip(prompts, assignments):
                    combined = PromptCleaner.cleanup_tags(f"{prompt_text}, {bg}")
                    updated.append(combined)
                prompts = updated

        return (prompts, civitai_id_output, character_tags, tags_rule, pixiv_tag)

class PackPromptGeneratorNode:
    """Node ComfyUI: Gerador de prompt de pack."""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.generator = PackPromptGeneratorCore(str(self.base_dir))

    @classmethod
    def _list_section_files(cls) -> List[str]:
        sections_dir = Path(__file__).parent / "sections"
        if sections_dir.exists():
            files = sorted([p.name for p in sections_dir.glob("*.txt")])
        else:
            files = []
        return files or ["section1.txt"]

    @classmethod
    def _list_spreadsheets(cls) -> List[str]:
        base_dir = Path(__file__).parent
        files = sorted([p.name for p in base_dir.glob("*.xlsx")])
        return files or ["nova_lista_fomatada.xlsx"]

    @classmethod
    def INPUT_TYPES(cls):
        section_files = cls._list_section_files()
        spreadsheet_files = cls._list_spreadsheets()
        default_s1 = "section1.txt" if "section1.txt" in section_files else section_files[0]
        default_s2 = "section2.txt" if "section2.txt" in section_files else section_files[0]
        default_s3 = "section3.txt" if "section3.txt" in section_files else section_files[0]
        default_sheet = "nova_lista_fomatada.xlsx" if "nova_lista_fomatada.xlsx" in spreadsheet_files else spreadsheet_files[0]

        return {
            "required": {
                "s1_source": (section_files, {"default": default_s1}),
                "s2_source": (section_files, {"default": default_s2}),
                "s3_source": (section_files, {"default": default_s3}),
                "spreadsheet": (spreadsheet_files, {"default": default_sheet}),
                "character_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Filtrar pelo TAGS RULE (ex: boku_no_hero)"
                }),
                "sequential_range": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Faixa sequencial (ex: 5~10)"
                }),
                "random_pool": (["all", "top 150", "top 100", "top 75", "top 50", "top 25"], {"default": "all"}),
                "civitai_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "ID/URL/AIR ID do Civitai (opcional)"
                }),
                "civitai_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "API key do Civitai (opcional)"
                }),
                "tags_rule_input": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Tags rule manual (opcional)"
                }),
                "s1_prompt_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "s2_prompt_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "s3_prompt_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "s1_outfit_count": ("INT", {"default": 1, "min": 0, "max": 20, "step": 1}),
                "s2_outfit_count": ("INT", {"default": 1, "min": 0, "max": 20, "step": 1}),
                "s3_outfit_count": ("INT", {"default": 1, "min": 0, "max": 20, "step": 1}),
                "background": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Backgrounds separados por / ou /cut (ex: home / castle, night)"
                }),
                "organizacao_inteligente": ("BOOLEAN", {"default": True}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Comentário do usuário para organização inteligente"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Você seleciona outfits de uma lista fornecida para três seções (s1, s2, s3). Retorne SOMENTE um JSON válido, sem explicações, no formato: {\"s1_outfit\": [...], \"s2_outfit\": [...], \"s3_outfit\": [...]}. Use apenas outfits da lista. Não invente.",
                    "placeholder": "System prompt para a AI (retorno em JSON)"
                }),
                "provider": (["openrouter", "groq"], {"default": "openrouter"}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "openai/gpt-4.1-nano"}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("generated_prompts", "civitai_id", "character_tags", "tags_rule", "pixiv_tag")
    OUTPUT_IS_LIST = (True, False, False, False, False)
    FUNCTION = "gerar"
    CATEGORY = "Prompt Utilities/Packreator"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def gerar(self,
              s1_source,
              s2_source,
              s3_source,
              spreadsheet,
              character_filter,
              sequential_range,
              random_pool,
              civitai_id,
              civitai_api_key,
              tags_rule_input,
              s1_prompt_count,
              s2_prompt_count,
              s3_prompt_count,
              s1_outfit_count,
              s2_outfit_count,
              s3_outfit_count,
              background,
              organizacao_inteligente,
              prompt,
              system_prompt,
              provider,
              api_key,
              model,
              pingpong,
              seed=-1):

        actual_seed = None if seed == -1 else seed

        try:
            return self.generator.gerar_prompts(
                spreadsheet=spreadsheet,
                s1_source=s1_source,
                s2_source=s2_source,
                s3_source=s3_source,
                character_filter=character_filter,
                sequential_range=sequential_range,
                random_pool=random_pool,
                civitai_id_input=civitai_id,
                civitai_api_key=civitai_api_key,
                tags_rule_input=tags_rule_input,
                s1_prompt_count=s1_prompt_count,
                s2_prompt_count=s2_prompt_count,
                s3_prompt_count=s3_prompt_count,
                s1_outfit_count=s1_outfit_count,
                s2_outfit_count=s2_outfit_count,
                s3_outfit_count=s3_outfit_count,
                background=background,
                organizacao_inteligente=organizacao_inteligente,
                prompt_usuario=prompt,
                system_prompt=system_prompt,
                provider=provider,
                api_key=api_key,
                model=model,
                pingpong=pingpong,
                seed=actual_seed,
            )
        except Exception as e:
            logger.error(f"Erro no Gerador de prompt de pack: {e}")
            return ([], "", "", f"Erro: {str(e)}", "")
