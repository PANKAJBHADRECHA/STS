# Third-party components

The direct native-audio worker follows the inference architecture/API illustrated by [Kyutai Moshi](https://github.com/kyutai-labs/moshi), whose code is MIT-licensed, copyright Kyutai. It is an independently packaged test harness, not a fork of the model or a redistributed checkpoint.

Kyutai's model card labels the selected `moshiko-mlx-q4` weights CC-BY-4.0. Model usage and attribution remain subject to that card. NVIDIA PersonaPlex weights use the NVIDIA Open Model License; no PersonaPlex weights are distributed here. Refer to exact upstream licenses before redistributing any model asset.

Optional dependencies are installed through their package distributions and retain their own licenses, including MLX, rustymimi, NumPy, sentencepiece and Playwright. Synthetic OS fixture generation uses macOS facilities locally; fixture audio is not committed.

MIT notice for Kyutai code patterns:

Copyright (c) Kyutai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
