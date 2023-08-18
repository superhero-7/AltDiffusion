export PYTHONPATH=$PYTHONPATH:/share/project/yfl/codebase/git/AltTools/Altdiffusion/src;
export CUDA_VISIBLE_DEVICES=7;
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;
# cp /share/project/yfl/codebase/data_connector.py /opt/conda/lib/python3.8/site-packages/pytorch_lightning/trainer/connectors/data_connector.py;
unset KUBERNETES_PORT;

# python -u /share/project/yfl/codebase/git/AltTools/Altdiffusion/src/scripts/train_multi.py --MASTER_ADDR $MASTER_ADDR --MASTER_PORT 29550 --WORLD_SIZE $WORLD_SIZE --NODE_RANK $RANK > /share/project/yfl/codebase/git/AltTools/Altdiffusion/src/log_txts/laion5plus_512_kv_xformer_cfg.txt
python -u /share/project/yfl/codebase/git/AltTools/Altdiffusion/src/scripts/train_multi.py --MASTER_ADDR $MASTER_ADDR --MASTER_PORT 29550 --WORLD_SIZE $WORLD_SIZE --NODE_RANK $RANK