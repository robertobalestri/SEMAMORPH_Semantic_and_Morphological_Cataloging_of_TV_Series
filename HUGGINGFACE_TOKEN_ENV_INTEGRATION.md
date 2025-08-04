# HuggingFace Token Environment Variable Integration

## Changes Made

✅ **Updated `whisperx_auth_token` property in `config.py`**:

**Before**:
```python
@property
def whisperx_auth_token(self) -> str:
    """HuggingFace authentication token for WhisperX."""
    return self.get_str('audio', 'auth_token', fallback='')
```

**After**:
```python
@property
def whisperx_auth_token(self) -> str:
    """HuggingFace authentication token for WhisperX."""
    # Try to get from environment variable first, then fallback to config.ini
    env_token = os.getenv('huggingface_auth_token', '').strip()
    if env_token:
        return env_token
    # Fallback to config.ini for backward compatibility
    return self.get_str('audio', 'auth_token', fallback='')
```

## Environment Variable Configuration

The token is now loaded from the `.env` file variable:
```properties
huggingface_auth_token = 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
```

## Benefits

🔐 **Environment-based Security**: Token stored in environment variables instead of config files
🔄 **Backward Compatibility**: Still falls back to `config.ini` if environment variable is not set
🚀 **Zero Code Changes Required**: All existing code continues to work as before
⚡ **Automatic Loading**: Token is automatically loaded via `load_dotenv()` at config initialization

## How It Works

1. **Environment Variable Priority**: The system first checks for `huggingface_auth_token` in environment variables
2. **Config.ini Fallback**: If the environment variable is empty or not set, it falls back to the `[audio]` section's `auth_token` in `config.ini`
3. **Integration**: All existing code using `config.whisperx_auth_token` automatically gets the environment variable value

## Usage Examples

The token is used throughout the audio processing pipeline:

- **Audio Processor**: `config.whisperx_auth_token`
- **Base Pipeline**: `config.get_auth_token()` → `config.whisperx_auth_token`
- **Audio Only Pipeline**: Uses the same configuration system
- **Pipeline Factory**: Validates token availability

All these components now automatically use the environment variable without any code changes required.
