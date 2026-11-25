# Enable Hugging Face / PersonaPlex access

The smaller LFM native voice model is public and does not require a token. PersonaPlex requires access under your own Hugging Face account.

1. Sign in at [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1), review and accept the model terms. Model access is tied to this account.
2. Open [Access Tokens](https://huggingface.co/settings/tokens). Create a **read** token, or a fine-grained token permitted to read the gated model. A token must belong to the account that accepted access; a token does not grant access by itself.
3. In Terminal, from this project, run:

   ```bash
   ./.venv-sts/bin/hf auth login
   ```

   Use the interactive login and enter the token there if requested. Do not paste it into chat, a source file or a command-line argument. Git credential storage is not needed for model downloads.
4. Check the login with `./.venv-sts/bin/hf auth whoami`. The app can then use Hugging Face's locally saved credential without you sharing the token in chat.

The login utility is already installed in this checkout's `.venv-sts`. For a fresh machine, install the native runtime first using the README instructions.

Sources: [Hugging Face CLI login](https://huggingface.co/docs/huggingface_hub/guides/cli), [token permissions](https://huggingface.co/docs/hub/security-tokens), [gated-model access](https://huggingface.co/docs/hub/models-gated).

Acceptance and authentication remove the access blocker; they do not make PersonaPlex's larger weights fit in 8 GB. The working voice application uses the smaller LFM model independently of PersonaPlex access.
