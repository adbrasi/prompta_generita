from .prompt_generator_node import PackPromptGeneratorNode

NODE_CLASS_MAPPINGS = {
    "PackPromptGenerator": PackPromptGeneratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PackPromptGenerator": "Gerador de prompt de pack"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("------------------------------------------")
print("PackCREATOR_BOLADEX Nodes LOADED")
print("------------------------------------------")
