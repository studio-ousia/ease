EASE_PATH=$PWD
DATA_PATH=$EASE_PATH/data
SENTEVA_PATH=$EASE_PATH/SentEval/data/downstream
CLT_PATH=$EASE_PATH/downstreams/cross-lingual-transfer/data

# download EASE dataset
mkdir -p $DATA_PATH
cd $DATA_PATH
wget https://huggingface.co/datasets/sosuke/dataset_for_ease/resolve/main/ease-dataset-en.json
wget https://huggingface.co/datasets/sosuke/dataset_for_ease/resolve/main/ease-dataset-18-langs.json

# download pretrained wikipedia2Vec embedding
S1="1vkV0eioNS4WX9R_CDTTyhCicvUEuaJu9";
S2="enwiki.fp16.768.vec";
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=$S1" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p');
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$S1" -O $S2;
rm -f /tmp/cookies.txt

# download STS dataset
mkdir -p $SENTEVA_PATH
cd $SENTEVA_PATH
wget https://huggingface.co/datasets/sosuke/dataset_for_ease/resolve/main/senteval.tar
tar xvf senteval.tar

# download MLDoc dataset
mkdir -p $CLT_PATH
cd $CLT_PATH
wget https://huggingface.co/datasets/sosuke/dataset_for_ease/resolve/main/mldoc.tar
tar xvf mldoc.tar