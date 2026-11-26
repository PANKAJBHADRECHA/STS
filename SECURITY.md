# Security policy

This is a research project, not a hardened multi-user voice service. Both application servers are intended to bind to loopback on a developer's machine. Do not expose them directly to the internet or treat model output as authorization to perform external actions.

## Reporting a vulnerability

If GitHub's **Report a vulnerability** option is available under this repository's Security tab, use it for private disclosure. Otherwise, open an issue requesting a private reporting channel without including exploit details, credentials, recordings or personal information. Do not post secrets in public issues.

There is no guaranteed response time or supported production release series. Security fixes target the current default branch.

## Credentials and audio

Selected application models are downloaded separately. Credentials, if needed for an optional gated-model experiment, belong in the Hugging Face credential store or an appropriate local secret store, never in source code or browser JavaScript.

Normal serving keeps conversation state in memory and does not write microphone recordings/transcripts. Streaming diagnostic mode is opt-in and can store private traces locally. Do not publish those traces without reviewing them. Browser/OS microphone permission is required; stopping a conversation should release capture.

A model license, an installed microphone permission, and a functioning model are separate concerns. None is a guarantee of safe or correct model behavior.
