#!/bin/bash -v

cd data

# get nl-De training data
wget -nc https://object.pouta.csc.fi/OPUS-CCMatrix/v1/moses/de-nl.txt.zip -O CCMatrix-de-nl.txt.zip
wget -nc https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/de-nl.txt.zip -O NLLB-de-nl.txt.zip
wget -nc https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/de-nl.txt.zip -O OpenSubtitles-de-nl.txt.zip
wget -nc https://object.pouta.csc.fi/OPUS-MultiParaCrawl/v9b/moses/de-nl.txt.zip -O MultiParaCrawl-de-nl.txt.zip
wget -nc https://object.pouta.csc.fi/OPUS-MultiCCAligned/v1.1/moses/de-nl.txt.zip -O MultiCCAligned-de-nl.txt.zip
wget -nc https://object.pouta.csc.fi/OPUS-ELRC-EMEA/v1/moses/de-nl.txt.zip -O ELRC-EMEA-de-nl.txt.zip
wget -nc https://object.pouta.csc.fi/OPUS-EUbookshop/v2/moses/de-nl.txt.zip -O EUbookshop-de-nl.txt.zip
wget -nc https://object.pouta.csc.fi/OPUS-DGT/v2019/moses/de-nl.txt.zip -O DGT-de-nl.txt.zip
wget -nc https://object.pouta.csc.fi/OPUS-TildeMODEL/v2018/moses/de-nl.txt.zip -O TildeMODEL-de-nl.txt.zip
wget -nc https://object.pouta.csc.fi/OPUS-LinguaTools-WikiTitles/v2014/moses/de-nl.txt.zip -O LinguaTools-WikiTitles-de-nl.txt.zip

# Unzip all files
unzip -o CCMatrix-de-nl.txt.zip
unzip -o NLLB-de-nl.txt.zip
unzip -o OpenSubtitles-de-nl.txt.zip
unzip -o MultiParaCrawl-de-nl.txt.zip
unzip -o MultiCCAligned-de-nl.txt.zip
unzip -o ELRC-EMEA-de-nl.txt.zip
unzip -o EUbookshop-de-nl.txt.zip
unzip -o DGT-de-nl.txt.zip
unzip -o TildeMODEL-de-nl.txt.zip
unzip -o LinguaTools-WikiTitles-de-nl.txt.zip

# Create corpus files
cat CCMatrix.de-nl.de NLLB.de-nl.de OpenSubtitles.de-nl.de MultiParaCrawl.de-nl.de MultiCCAligned.de-nl.de ELRC-EMEA.de-nl.de EUbookshop.de-nl.de DGT.de-nl.de TildeMODEL.de-nl.de LinguaTools-WikiTitles.de-nl.de > corpus-ordered.de
cat CCMatrix.de-nl.nl NLLB.de-nl.nl OpenSubtitles.de-nl.nl MultiParaCrawl.de-nl.nl MultiCCAligned.de-nl.nl ELRC-EMEA.de-nl.nl EUbookshop.de-nl.nl DGT.de-nl.nl TildeMODEL.de-nl.nl LinguaTools-WikiTitles.de-nl.nl > corpus-ordered.nl

# Shuffle the corpus
shuf --random-source=corpus-ordered.de corpus-ordered.de > corpus-full.de
shuf --random-source=corpus-ordered.de corpus-ordered.nl > corpus-full.nl

# Make data splits
head -n -4000 corpus-full.de > corpus.de
head -n -4000 corpus-full.nl > corpus.nl
tail -n 4000 corpus-full.de > corpus-dev-test.de
tail -n 4000 corpus-full.nl > corpus-dev-test.nl
head -n 2000 corpus-dev-test.de > corpus-dev.de
head -n 2000 corpus-dev-test.nl > corpus-dev.nl
tail -n 2000 corpus-dev-test.de > corpus-test.de
tail -n 2000 corpus-dev-test.nl > corpus-test.nl

cd ..