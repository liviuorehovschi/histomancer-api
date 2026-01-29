#!/bin/sh
# One-time push to HF Space. Requires HF_TOKEN and HF_USERNAME in environment.
# e.g. HF_TOKEN=hf_xxx HF_USERNAME=liviuorehovschi ./push-to-hf.sh
set -e
HF_USERNAME=${HF_USERNAME:-liviuorehovschi}
git push "https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/histomancer-api" main
