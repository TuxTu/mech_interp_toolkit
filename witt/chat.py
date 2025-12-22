"""
Chat storage classes for the witt library.
"""
from typing import Optional, List, Tuple

from .prompt import Prompt


class ChatRoleList:
    """A list wrapper for prompts in a specific chat role (user/assistant)."""
    
    def __init__(self, prompts: List[Prompt]):
        self._prompts = prompts
    
    def __getitem__(self, idx: int) -> Prompt:
        """Access prompt by index (supports negative indexing)."""
        return self._prompts[idx]
    
    def __len__(self) -> int:
        return len(self._prompts)
    
    def __iter__(self):
        return iter(self._prompts)
    
    def __repr__(self) -> str:
        return f"ChatRoleList({len(self._prompts)} prompts)"


class Chat:
    """
    Represents a chat conversation with system, user, and assistant prompts.
    
    Structure:
        chat["system"] -> single Prompt for system message
        chat["user"] -> ChatRoleList of user prompts
        chat["assistant"] -> ChatRoleList of assistant prompts
    
    Access pattern for token-level operations:
        chat["system"][token_idx][layer_idx][module] = value
        chat["user"][prompt_idx][token_idx][layer_idx][module] = value
        chat["assistant"][prompt_idx][token_idx][layer_idx][module] = value
    
    Example:
        chat = Chat()
        chat.set_system("You are a helpful assistant.")
        chat.add_user("Hello!")
        chat.add_assistant("Hi there!")
        
        # Access and patch activations
        chat["user"][0][-1][10]["resid_post"] = some_value
    """
    
    def __init__(self):
        self._system: Optional[Prompt] = None
        self._user: List[Prompt] = []
        self._assistant: List[Prompt] = []
        self._prompt_counter: int = 0  # Track unique prompt IDs across all roles
    
    def set_system(self, text: str, tokens: Optional[List[Tuple[int, str]]] = None) -> Prompt:
        """Set the system prompt (replaces existing if any)."""
        prompt = Prompt(text, id=self._prompt_counter, tokens=tokens)
        self._system = prompt
        self._prompt_counter += 1
        return prompt
    
    def add_user(self, text: str, tokens: Optional[List[Tuple[int, str]]] = None) -> Prompt:
        """Add a user prompt to the conversation."""
        prompt = Prompt(text, id=self._prompt_counter, tokens=tokens)
        self._user.append(prompt)
        self._prompt_counter += 1
        return prompt
    
    def add_assistant(self, text: str, tokens: Optional[List[Tuple[int, str]]] = None) -> Prompt:
        """Add an assistant prompt to the conversation."""
        prompt = Prompt(text, id=self._prompt_counter, tokens=tokens)
        self._assistant.append(prompt)
        self._prompt_counter += 1
        return prompt
    
    def __getitem__(self, role: str):
        """
        Access prompts by role.
        
        Args:
            role: One of 'system', 'user', or 'assistant'
            
        Returns:
            For 'system': the Prompt directly (or raises KeyError if not set)
            For 'user'/'assistant': a ChatRoleList for indexing into prompts
        """
        if role == "system":
            if self._system is None:
                raise KeyError("No system prompt set")
            return self._system
        elif role == "user":
            return ChatRoleList(self._user)
        elif role == "assistant":
            return ChatRoleList(self._assistant)
        else:
            raise KeyError(f"Unknown role: {role}. Valid roles: 'system', 'user', 'assistant'")
    
    @property
    def system(self) -> Optional[Prompt]:
        """Direct access to system prompt."""
        return self._system
    
    @property
    def user(self) -> ChatRoleList:
        """Direct access to user prompts list."""
        return ChatRoleList(self._user)
    
    @property
    def assistant(self) -> ChatRoleList:
        """Direct access to assistant prompts list."""
        return ChatRoleList(self._assistant)
    
    def __repr__(self) -> str:
        system_str = "1 system" if self._system else "no system"
        return f"Chat({system_str}, {len(self._user)} user, {len(self._assistant)} assistant)"
    
    def __len__(self) -> int:
        """Total number of prompts in the chat."""
        count = len(self._user) + len(self._assistant)
        if self._system:
            count += 1
        return count

