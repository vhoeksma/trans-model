#!/bin/bash -v

MARIAN=../..

# Set source and target languages
SRC=nl  # Dutch
TGT=de  # German

# Set chosen GPUs
GPUS=0
if [ $# -ne 0 ]; then
    GPUS=$@
fi
echo "Using GPUs: $GPUS"

if [ ! -e $MARIAN/build/marian ]; then
    echo "Marian is not installed in $MARIAN/build, you need to compile the toolkit first"
    exit 1
fi

# List of download links for training data
DATA_URLS=(
    "https://object.pouta.csc.fi/OPUS-CCMatrix/v1/moses/de-nl.txt.zip CCMatrix-de-nl.txt.zip"
    "https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/de-nl.txt.zip NLLB-de-nl.txt.zip"
    "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/de-nl.txt.zip OpenSubtitles-de-nl.txt.zip"
    "https://object.pouta.csc.fi/OPUS-MultiParaCrawl/v9b/moses/de-nl.txt.zip MultiParaCrawl-de-nl.txt.zip"
    "https://object.pouta.csc.fi/OPUS-MultiCCAligned/v1.1/moses/de-nl.txt.zip MultiCCAligned-de-nl.txt.zip"
    "https://object.pouta.csc.fi/OPUS-ELRC-EMEA/v1/moses/de-nl.txt.zip ELRC-EMEA-de-nl.txt.zip"
    "https://object.pouta.csc.fi/OPUS-EUbookshop/v2/moses/de-nl.txt.zip EUbookshop-de-nl.txt.zip"
    "https://object.pouta.csc.fi/OPUS-DGT/v2019/moses/de-nl.txt.zip DGT-de-nl.txt.zip"
    "https://object.pouta.csc.fi/OPUS-TildeMODEL/v2018/moses/de-nl.txt.zip TildeMODEL-de-nl.txt.zip"
    "https://object.pouta.csc.fi/OPUS-LinguaTools-WikiTitles/v2014/moses/de-nl.txt.zip LinguaTools-WikiTitles-de-nl.txt.zip"
)

# Create data directory if it does not exist
mkdir -p data
cd data

# Download training data (only if not already downloaded)
for ENTRY in "${DATA_URLS[@]}"; do
    URL=$(echo $ENTRY | awk '{print $1}')
    FILE_NAME=$(echo $ENTRY | awk '{print $2}')
    
    # Check if file is already downloaded, if not, download
    if [ ! -e "$FILE_NAME" ]; then
        echo "Downloading $FILE_NAME..."
        wget -nc "$URL" -O "$FILE_NAME"
    else
        echo "$FILE_NAME already exists, skipping download."
    fi
    
    # Check if the extracted file(s) exist, if not, extract
    if [ ! -e "${FILE_NAME%.zip}" ]; then
        echo "Extracting $FILE_NAME..."
        unzip -o "$FILE_NAME"
    else
        echo "Extracted files for $FILE_NAME already exist, skipping extraction."
    fi
done

# Concatenate corpus files
cat *.$SRC > corpus.$SRC
cat *.$TGT > corpus.$TGT

# Clean up the extracted .zip files (keeping them as you requested)
# rm -f *.zip  # Don't delete the zip files, as you want to keep them

# Change back to main directory
cd ..

# Get our fork of sacreBLEU
git clone https://github.com/marian-nmt/sacreBLEU.git sacreBLEU

# Manually set up dev and test sets from the corpus data
echo "Manually setting up validation and test sets..."

# Example: Use the first 1000 lines of the corpus as the validation set
head -n 1000 data/corpus.$SRC > data/transfile.$SRC
head -n 1000 data/corpus.$TGT > data/transfile.$TGT

# Example: Use the next 1000 lines as the test set
tail -n +1001 data/corpus.$SRC | head -n 1000 > data/newstest2016.$SRC
tail -n +1001 data/corpus.$TGT | head -n 1000 > data/newstest2016.$TGT

# Create the model folder
mkdir -p model

# Train the model
$MARIAN/build/marian \
    --devices $GPUS \
    --type s2s \
    --model model/model.npz \
    --train-sets data/corpus.$SRC data/corpus.$TGT \
    --vocabs model/vocab.$SRC$TGT.spm model/vocab.$SRC$TGT.spm \
    --dim-vocabs 32000 32000 \
    --mini-batch-fit -w 2000 \
    --layer-normalization --tied-embeddings-all \
    --dropout-rnn 0.2 --dropout-src 0.1 --dropout-trg 0.1 \
    --early-stopping 5 --max-length 100 \
    --valid-freq 10000 --save-freq 10000 --disp-freq 1000 \
    --cost-type ce-mean-words --valid-metrics ce-mean-words bleu-detok \
    --valid-sets data/transfile.$SRC data/transfile.$TGT \
    --log model/train.log --valid-log model/valid.log --tempdir model \
    --overwrite --keep-best \
    --seed 1111 --exponential-smoothing \
    --normalize=0.6 --beam-size=6 --quiet-translation

# Translate dev set
cat data/transfile.$SRC \
    | $MARIAN/build/marian-decoder -c model/model.npz.best-bleu-detok.npz.decoder.yml -d $GPUS -b 6 -n0.6 \
      --mini-batch 32 --maxi-batch 100 --maxi-batch-sort src > data/transfile.$SRC.output

# Translate test set
cat data/newstest2016.$SRC \
    | $MARIAN/build/marian-decoder -c model/model.npz.best-bleu-detok.npz.decoder.yml -d $GPUS -b 6 -n0.6 \
      --mini-batch 32 --maxi-batch 100 --maxi-batch-sort src > data/newstest2016.$SRC.output

# Calculate BLEU scores
sacreBLEU/sacrebleu.py -t wmt16/dev -l $SRC-$TGT < data/transfile.$SRC.output
sacreBLEU/sacrebleu.py -t wmt16 -l $SRC-$TGT < data/newstest2016.$SRC.output
