# System Patterns

This file documents recurring patterns and standards used in the project.
It is optional, but recommended to be updated as the project evolves.

2025-01-02 22:32:00 - Initial system patterns documentation based on project analysis.

## Coding Patterns

- **Modular Architecture**: Separate modules for different AI models (`wan/`, `hyvideo/`, `ltx_video/`)
- **Configuration-Driven**: Extensive use of configuration files and command-line parameters
- **Resource Management**: Memory-aware loading patterns for different hardware profiles
- **Web Interface**: Gradio-based UI with consistent parameter handling across models
- **Plugin Architecture**: LoRA support with dynamic loading/unloading capabilities

## Architectural Patterns

- **Adapter Pattern**: Unified interface for different video generation models
- **Strategy Pattern**: Multiple attention mechanisms (SDPA, Sage, Flash) with runtime selection
- **Factory Pattern**: Model instantiation based on configuration and hardware capabilities
- **Observer Pattern**: Progress tracking and status updates during generation
- **Command Pattern**: Queuing system for batch video generation
- **Facade Pattern**: Simplified API hiding complexity of underlying model implementations

## Testing Patterns

- **Hardware Compatibility Testing**: Multi-GPU architecture validation
- **Performance Benchmarking**: VRAM usage and generation speed optimization
- **Model Integration Testing**: Cross-model compatibility and feature validation
- **User Interface Testing**: Web interface functionality across different browsers
- **Resource Constraint Testing**: Low-VRAM scenario validation