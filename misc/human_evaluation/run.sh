# languages=("ar" "de" "es" "fr" "hi" "it" "ja" "nl" "ru" "th" "tr" "zh" "uk" "pl" "ko" "pt" "vi")
CUDA_VISIBLE_DEVICES=0 python stable_diffusion_api.py --lg1 "ar" --lg2 "de"&
CUDA_VISIBLE_DEVICES=1 python stable_diffusion_api.py --lg1 "es" --lg2 "fr"&
CUDA_VISIBLE_DEVICES=2 python stable_diffusion_api.py --lg1 "hi" --lg2 "it"&
CUDA_VISIBLE_DEVICES=3 python stable_diffusion_api.py --lg1 "ja" --lg2 "nl"&
wait
CUDA_VISIBLE_DEVICES=0 python stable_diffusion_api.py --lg1 "ru" --lg2 "th"&
CUDA_VISIBLE_DEVICES=1 python stable_diffusion_api.py --lg1 "tr" --lg2 "zh"&
CUDA_VISIBLE_DEVICES=2 python stable_diffusion_api.py --lg1 "uk" --lg2 "pl"&
CUDA_VISIBLE_DEVICES=3 python stable_diffusion_api.py --lg1 "ko" --lg2 "pt"