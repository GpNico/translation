#!/bin/bash
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

#
# Data preprocessing configuration
#

N_MONO=100000  # number of monolingual sentences for each language
CODES=1000      # number of BPE codes
N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=30      # number of fastText epochs
SRC_NAME=000000
TGT_NAME=000000
LEXICON_SRC=0
LEXICON_TGT=0
FREQ_SRC=0 # If 0 then uniform freq, else it correspond to the k*10 of the power law (so k = FREQ/10) # Because FREQ cannot be a float :(
FREQ_TGT=0 
PARA=False # the target is now parallel to the source
FIELD=False # download lexical field data /!\ Freq is k=1.1, no need to specify FREQ_XXX /!\

# Name of the experiment
SRC_STRING='s'$SRC_NAME'.LEX'$LEXICON_SRC
TGT_STRING='t'$TGT_NAME'.LEX'$LEXICON_TGT

if [ $FREQ_SRC -ge 1 ];
then
  SRC_STRING=$SRC_STRING'.FREQ_K'$FREQ_SRC 
fi

if [ $FREQ_TGT -ge 1 ];
then
  TGT_STRING=$TGT_STRING'.FREQ_K'$FREQ_TGT 
fi


EXP_NAME='GR_'$SRC_STRING'.'$TGT_STRING

if [ $FIELD = True ];
then
  echo "Download lexical field data..."
  EXP_NAME=$EXP_NAME'_FIELD'
fi

if [ $PARA = True ];
then
  echo "Parallel data..."
  EXP_NAME=$EXP_NAME'_SUP'
fi



#
# Initialize tools and data paths
#

# main paths
UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
GRAMMARS_PATH=$DATA_PATH/artificial_grammars

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $DATA_PATH
mkdir -p $GRAMMARS_PATH
#mkdir -p $GRAMMARS_PATH/source
#mkdir -p $GRAMMARS_PATH/valid
#mkdir -p $GRAMMARS_PATH/test
mkdir -p $GRAMMARS_PATH/$EXP_NAME
mkdir -p $GRAMMARS_PATH/$EXP_NAME/valid
mkdir -p $GRAMMARS_PATH/$EXP_NAME/test
mkdir -p $GRAMMARS_PATH/$EXP_NAME/valid/valid_src
mkdir -p $GRAMMARS_PATH/$EXP_NAME/valid/valid_tgt
mkdir -p $GRAMMARS_PATH/$EXP_NAME/test/test_src
mkdir -p $GRAMMARS_PATH/$EXP_NAME/test/test_tgt
mkdir -p $GRAMMARS_PATH/$EXP_NAME/source
mkdir -p $GRAMMARS_PATH/$EXP_NAME/target

# moses
MOSES=$TOOLS_PATH/mosesdecoder
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
#INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

# files full paths
SRC_RAW=$GRAMMARS_PATH/$EXP_NAME/source/sample_$SRC_NAME.txt
SRC_TOK=$GRAMMARS_PATH/$EXP_NAME/sample_s$SRC_NAME.LEX$LEXICON_SRC.tok # Always lexicon 0 as source
TGT_TOK=$GRAMMARS_PATH/$EXP_NAME/sample_t$TGT_NAME.LEX$LEXICON_TGT.tok
  
BPE_CODES=$GRAMMARS_PATH/$EXP_NAME/bpe_codes$CODES.s$SRC_NAME.LEX$LEXICON_SRC-t$TGT_NAME.LEX$LEXICON_TGT

CONCAT_BPE=$GRAMMARS_PATH/$EXP_NAME/all.s$SRC_NAME.LEX$LEXICON_SRC-t$TGT_NAME.LEX$LEXICON_TGT.$CODES

SRC_VOCAB=$GRAMMARS_PATH/$EXP_NAME/vocab.s$SRC_NAME.LEX$LEXICON_SRC.$CODES
TGT_VOCAB=$GRAMMARS_PATH/$EXP_NAME/vocab.t$TGT_NAME.LEX$LEXICON_TGT.$CODES
FULL_VOCAB=$GRAMMARS_PATH/$EXP_NAME/vocab.s$SRC_NAME.LEX$LEXICON_SRC-t$TGT_NAME.LEX$LEXICON_TGT.$CODES

SRC_VALID=$GRAMMARS_PATH/$EXP_NAME/valid/valid_src/sample_$SRC_NAME.txt
SRC_TEST=$GRAMMARS_PATH/$EXP_NAME/test/test_src/sample_$SRC_NAME.txt

SRC_VALID_TOK=$GRAMMARS_PATH/$EXP_NAME/valid/sample_s$SRC_NAME.LEX$LEXICON_SRC.tok
SRC_TEST_TOK=$GRAMMARS_PATH/$EXP_NAME/test/sample_s$SRC_NAME.LEX$LEXICON_SRC.tok
TGT_VALID_TOK=$GRAMMARS_PATH/$EXP_NAME/valid/sample_t$TGT_NAME.LEX$LEXICON_TGT.tok
TGT_TEST_TOK=$GRAMMARS_PATH/$EXP_NAME/test/sample_t$TGT_NAME.LEX$LEXICON_TGT.tok

### Modification ###
TGT_RAW=$GRAMMARS_PATH/$EXP_NAME/target/sample_$TGT_NAME.txt
TGT_VALID=$GRAMMARS_PATH/$EXP_NAME/valid/valid_tgt/sample_$TGT_NAME.txt
TGT_TEST=$GRAMMARS_PATH/$EXP_NAME/test/test_tgt/sample_$TGT_NAME.txt

#
# Download and install tools
#

# Download Moses
cd $TOOLS_PATH
if [ ! -d "$MOSES" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
fi
echo "Moses found in: $MOSES"

# Download fastBPE
cd $TOOLS_PATH
if [ ! -d "$FASTBPE_DIR" ]; then
  echo "Cloning fastBPE from GitHub repository..."
  git clone https://github.com/glample/fastBPE
fi
echo "fastBPE found in: $FASTBPE_DIR"

# Compile fastBPE
cd $TOOLS_PATH
if [ ! -f "$FASTBPE" ]; then
  echo "Compiling fastBPE..."
  cd $FASTBPE_DIR
  g++ -std=c++11 -pthread -O3 fastBPE/main.cc -o fast
fi
echo "fastBPE compiled in: $FASTBPE"

# Download fastText
cd $TOOLS_PATH
if [ ! -d "$FASTTEXT_DIR" ]; then
  echo "Cloning fastText from GitHub repository..."
  git clone https://github.com/facebookresearch/fastText.git
fi
echo "fastText found in: $FASTTEXT_DIR"

# Compile fastText
cd $TOOLS_PATH
if [ ! -f "$FASTTEXT" ]; then
  echo "Compiling fastText..."
  cd $FASTTEXT_DIR
  make
fi
echo "fastText compiled in: $FASTTEXT"


#
# Download monolingual data
#
if ! [[ -f "$SRC_RAW" ]]; then
  echo "Installing gdown..."
  
  pip install gdown

  echo "Downloading Artificial Grammars..."

  cd $GRAMMARS_PATH/$EXP_NAME/source
  
  if [ $FIELD = True ];
  then
    gdown https://drive.google.com/uc?id=1SnUriwTshjqkXAbBr-XJtqBOksZO7BXA
    unzip -j freqk11_permuted_samples_source_fields.zip
  elif [ $FREQ_SRC -eq 11 ];
  then
    gdown https://drive.google.com/uc?id=13CqOmZZ7PtcXfb11Me91Yye2KeEVBwkC
    unzip -j freqk11_permuted_samples_source.zip
  elif [ $FREQ_SRC -eq 20 ];
  then
    gdown https://drive.google.com/uc?id=1iM2L1RyFXt5zjQN2Udrr8uuG73JYnYlj
    unzip -j freqk2_permuted_samples_source.zip
  else
    gdown https://drive.google.com/uc?id=1tNzIr3JHe0JRXFa87qGnG46V-QrF7dJQ
    unzip -j permuted_samples_source.zip
  fi
fi
 
if ! [[ -f "$TGT_RAW" ]]; then
  
  cd $GRAMMARS_PATH/$EXP_NAME/target
  
  if [ $LEXICON_TGT -eq 0 ];
  then
    if [ $PARA = True ];
    then
      if [ $FIELD = True ];
      then
        gdown https://drive.google.com/uc?id=1SnUriwTshjqkXAbBr-XJtqBOksZO7BXA
        unzip -j freqk11_permuted_samples_source_fields.zip
      elif [ $FREQ_TGT -eq 11 ];
      then
        gdown https://drive.google.com/uc?id=13CqOmZZ7PtcXfb11Me91Yye2KeEVBwkC
        unzip -j freqk11_permuted_samples_source.zip
      elif [ $FREQ_TGT -eq 20 ];
      then
        gdown https://drive.google.com/uc?id=1iM2L1RyFXt5zjQN2Udrr8uuG73JYnYlj
        unzip -j freqk2_permuted_samples_source.zip
      else
        gdown https://drive.google.com/uc?id=1tNzIr3JHe0JRXFa87qGnG46V-QrF7dJQ
        unzip -j permuted_samples_source.zip
      fi
    else
      if [ $FIELD = True ];
      then
        gdown https://drive.google.com/uc?id=1KoAqglDnQOmu8oToJyvWhT6cEnVCqkfA
        unzip -j freqk11_permuted_samples_target_fields.zip
      elif [ $FREQ_TGT -eq 11 ];
      then
        gdown https://drive.google.com/uc?id=1NwWyP3_stFjdpN2-yNRQ1YAjCOu9xtX-
        unzip -j freqk11_permuted_samples_target.zip
      elif [ $FREQ_TGT -eq 20 ];
      then
        gdown https://drive.google.com/uc?id=1_lvzmqa_hvlBDeiDSZFiew8b6Jd7xYlw
        unzip -j freqk2_permuted_samples_target.zip
      else
        gdown https://drive.google.com/uc?id=1ak6eXWB054Y3Zg3wQc0n2zjEFdLW4-cm
        unzip -j permuted_samples_target.zip
      fi
    fi
    
  fi
  if [ $LEXICON_TGT -eq 1 ];
  then
    
    if [ $PARA = True ];
    then
      if [ $FIELD = True ];
      then
        gdown https://drive.google.com/uc?id=1mdSNy_B1SwWAPxAZmnnhJgMtga8Oi1iK
        unzip -j freqk11_permuted_samples_source_fields_lexicon_1.zip
      elif [ $FREQ_TGT -eq 11 ];
      then
        gdown https://drive.google.com/uc?id=17UZ5daDavOKHAiEQx_6KDQYSqE2n7BJ4
        unzip -j freqk11_permuted_samples_source_lexicon_1.zip
      elif [ $FREQ_TGT -eq 20 ];
      then
        gdown https://drive.google.com/uc?id=1Rck1Rnzh213yrDe2vAgUzR0Gs9mHDwjG
        unzip -j freqk2_permuted_samples_source_lexicon_1.zip
      else
        gdown https://drive.google.com/uc?id=1XM8rPKrEePsJPZYsV_bv7pAF1Iv6pHGz
        unzip -j permuted_samples_source_lexicon_1.zip
      fi
    else
      if [ $FIELD = True ];
      then
        gdown https://drive.google.com/uc?id=1X-2Wip7SMeyBxKGUfSiRoBjPZrwL3wpN
        unzip -j freqk11_permuted_samples_target_fields_lexicon_1.zip
      elif [ $FREQ_TGT -eq 11 ];
      then
        gdown https://drive.google.com/uc?id=1lByQHzaUwuKxSHU6Yz67DJpAbTF2MCtG
        unzip -j freqk11_permuted_samples_target_lexicon_1.zip
      elif [ $FREQ_TGT -eq 20 ];
      then
        gdown https://drive.google.com/uc?id=1QRofOl2QaHuZTjCDmyEnOTo74rUuZrz-
        unzip -j freqk2_permuted_samples_target_lexicon_1.zip
      else
        gdown https://drive.google.com/uc?id=16piuveQXb0dACvGMA8PD_aH7X5JSKOyz
        unzip -j permuted_samples_target_lexicon_1.zip
      fi
    fi
  fi
  if [ $LEXICON_TGT -eq 2 ];
  then
    
    if [ $PARA = True ];
    then
      gdown https://drive.google.com/uc?id=1gF6tv5AradQn1AW2HQzQBZf7VZef6aaH
      unzip -j permuted_samples_source_lexicon_2.zip
    else
      gdown https://drive.google.com/uc?id=1JCxht0OR_ZVCU70nxQLYfDBLKNdRm6Ab
      unzip -j permuted_samples_target_lexicon_2.zip
    fi
  fi
fi


# concatenate monolingual data files
#if ! [[ -f "$SRC_RAW" && -f "$TGT_RAW" ]]; then
#  echo "Concatenating monolingual data..."
#  cat $(ls news*en* | grep -v gz) | head -n $N_MONO > $SRC_RAW
#  cat $(ls news*fr* | grep -v gz) | head -n $N_MONO > $TGT_RAW
#fi
echo "Grammar $SRC_NAME monolingual data concatenated in: $SRC_RAW"
echo "Grammar $TGT_NAME monolingual data concatenated in: $TGT_RAW"

# check number of lines
if ! [[ "$(wc -l < $SRC_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your $SRC_NAME monolingual data."; exit; fi
if ! [[ "$(wc -l < $TGT_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your $TGT_NAME monolingual data."; exit; fi

# tokenize data
if ! [[ -f "$SRC_TOK" && -f "$TGT_TOK" ]]; then
  echo "Tokenize monolingual data..."
  cat $SRC_RAW | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TOK
  cat $TGT_RAW | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $TGT_TOK
fi
echo "$SRC_NAME monolingual data tokenized in: $SRC_TOK"
echo "$TGT_NAME monolingual data tokenized in: $TGT_TOK"

# learn BPE codes
if [ ! -f "$BPE_CODES" ]; then
  echo "Learning BPE codes..."
  $FASTBPE learnbpe $CODES $SRC_TOK $TGT_TOK > $BPE_CODES
fi
echo "BPE learned in $BPE_CODES"

# apply BPE codes
if ! [[ -f "$SRC_TOK.$CODES" && -f "$TGT_TOK.$CODES" ]]; then
  echo "Applying BPE codes..."
  $FASTBPE applybpe $SRC_TOK.$CODES $SRC_TOK $BPE_CODES
  $FASTBPE applybpe $TGT_TOK.$CODES $TGT_TOK $BPE_CODES
fi
echo "BPE codes applied to $SRC_NAME in: $SRC_TOK.$CODES"
echo "BPE codes applied to $TGT_NAME in: $TGT_TOK.$CODES"

# extract vocabulary
if ! [[ -f "$SRC_VOCAB" && -f "$TGT_VOCAB" && -f "$FULL_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $SRC_TOK.$CODES > $SRC_VOCAB
  $FASTBPE getvocab $TGT_TOK.$CODES > $TGT_VOCAB
  $FASTBPE getvocab $SRC_TOK.$CODES $TGT_TOK.$CODES > $FULL_VOCAB
fi
echo "$SRC_NAME vocab in: $SRC_VOCAB"
echo "$TGT_NAME vocab in: $TGT_VOCAB"
echo "Full vocab in: $FULL_VOCAB"

# binarize data
if ! [[ -f "$SRC_TOK.$CODES.pth" && -f "$TGT_TOK.$CODES.pth" ]]; then
  echo "Binarizing data..."
  $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TOK.$CODES
  $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TOK.$CODES
fi
echo "$SRC_NAME binarized data in: $SRC_TOK.$CODES.pth"
echo "$TGT_NAME binarized data in: $TGT_TOK.$CODES.pth"

#
# Download parallel data (for evaluation only)
#

if ! [[ -f "$SRC_VALID" ]]; then
  echo "Downloading Valid data..."
  cd $GRAMMARS_PATH/$EXP_NAME/valid/valid_src
  
  if [ $FIELD = True ];
  then
    gdown https://drive.google.com/uc?id=18DzgZqvvIs5rYJeqpgbUfO_3B42T7oAb
    unzip -j freqk11_permuted_samples_valid_fields.zip
  elif [ $FREQ_TGT -eq 11 ];
  then
    gdown https://drive.google.com/uc?id=1gJNqMuVznjHjcObXiqwxk-7t-6RMgjwn
    unzip -j freqk11_permuted_samples_valid.zip
  elif [ $FREQ_TGT -eq 20 ];
  then
    gdown https://drive.google.com/uc?id=12YrU6YTMnjdpzxfu9esdwLUb51i5pMAG
    unzip -j freqk2_permuted_samples_valid.zip
  else
    gdown https://drive.google.com/uc?id=10_j0MK8jiOe8C0JvAuPz-Dw5Zm_Lqkn3
    unzip -j permuted_samples_valid.zip
  fi
fi

if ! [[ -f "$TGT_VALID" ]]; then
  cd $GRAMMARS_PATH/$EXP_NAME/valid/valid_tgt
  if [ $LEXICON_TGT -eq 0 ];
  then
    if [ $FIELD = True ];
    then
      gdown https://drive.google.com/uc?id=18DzgZqvvIs5rYJeqpgbUfO_3B42T7oAb
      unzip -j freqk11_permuted_samples_valid_fields.zip
    elif [ $FREQ_TGT -eq 11 ];
    then
      gdown https://drive.google.com/uc?id=1gJNqMuVznjHjcObXiqwxk-7t-6RMgjwn
      unzip -j freqk11_permuted_samples_valid.zip
    elif [ $FREQ_TGT -eq 20 ];
    then
      gdown https://drive.google.com/uc?id=12YrU6YTMnjdpzxfu9esdwLUb51i5pMAG
      unzip -j freqk2_permuted_samples_valid.zip
    else
      gdown https://drive.google.com/uc?id=10_j0MK8jiOe8C0JvAuPz-Dw5Zm_Lqkn3
      unzip -j permuted_samples_valid.zip
    fi
  elif [ $LEXICON_TGT -eq 1 ];
  then
    if [ $FIELD = True ];
    then
      gdown https://drive.google.com/uc?id=1r-mT7oneUj9tEGHL6lrOe1DZU5JovUL7
      unzip -j freqk11_permuted_samples_valid_fields_lexicon_1.zip
    elif [ $FREQ_TGT -eq 11 ];
    then
      gdown https://drive.google.com/uc?id=1rtBYeyBN86tu1U3wak6KZMaoRJ0_Sgc-
      unzip -j freqk11_permuted_samples_valid_lexicon_1.zip
    elif [ $FREQ_TGT -eq 20 ];
    then
      gdown https://drive.google.com/uc?id=1TOYIkrgJs-bsORR-vk2OD3Wg2ajHSOPi
      unzip -j freqk2_permuted_samples_valid_lexicon_1.zip
    else
      gdown https://drive.google.com/uc?id=17bcTIJA8lNyZDhdBKhAef2FCwstgCB12
      unzip -j permuted_samples_valid_lexicon_1.zip
    fi
  elif [ $LEXICON_TGT -eq 2 ];
  then
    gdown https://drive.google.com/uc?id=15jG-1GOBRW1upzMrW6vivoC1ZT32ZDVm
    unzip -j permuted_samples_valid_lexicon_2.zip
  fi
fi

if ! [[ -f "$SRC_TEST" ]]; then
  echo "Downloading Test data..."

  cd $GRAMMARS_PATH/$EXP_NAME/test/test_src
  
  if [ $FIELD = True ];
  then
    gdown https://drive.google.com/uc?id=1WAUw1HM7Va_HT15OCDsrDFT6sk-IEd4Z
    unzip -j freqk11_permuted_samples_test_fields.zip
  elif [ $FREQ_TGT -eq 11 ];
  then
    gdown https://drive.google.com/uc?id=1eIoZJ5tb19H0prDcaVwx1nI936onRnoQ
    unzip -j freqk11_permuted_samples_test.zip
  elif [ $FREQ_TGT -eq 20 ];
  then
    gdown https://drive.google.com/uc?id=1msZmwX6jEXyMz7ukOI2QkcxQwOhefXNk
    unzip -j freqk2_permuted_samples_test.zip
  else
    gdown https://drive.google.com/uc?id=1zd_fL7RIfCp8YE9zz8tMIbUuOiBROd8P
    unzip -j permuted_samples_test.zip
  fi
fi

echo "We reach line 456"

if ! [[ -f "$TGT_TEST" ]]; then
  echo "We are in TYGT_TEST"
  cd $GRAMMARS_PATH/$EXP_NAME/test/test_tgt
  echo "We cd-ed"
  if [ $LEXICON_TGT -eq 0 ];
  then
    echo "We are in tgt_test lex0"
    if [ $FIELD = True ];
    then
      gdown https://drive.google.com/uc?id=1WAUw1HM7Va_HT15OCDsrDFT6sk-IEd4Z
      unzip -j freqk11_permuted_samples_test_fields.zip
    elif [ $FREQ_TGT -eq 11 ];
    then
      gdown https://drive.google.com/uc?id=1eIoZJ5tb19H0prDcaVwx1nI936onRnoQ
      unzip -j freqk11_permuted_samples_test.zip
    elif [ $FREQ_TGT -eq 20 ];
    then
      gdown https://drive.google.com/uc?id=1msZmwX6jEXyMz7ukOI2QkcxQwOhefXNk
      unzip -j freqk2_permuted_samples_test.zip
    else
      echo "We are in permuted_samples_test.zip"
      gdown https://drive.google.com/uc?id=1zd_fL7RIfCp8YE9zz8tMIbUuOiBROd8P
      unzip -j permuted_samples_test.zip
    fi
  elif [ $LEXICON_TGT -eq 1 ];
  then
    if [ $FIELD = True ];
    then
      gdown https://drive.google.com/uc?id=1PYoT2nhSApJ7HXXxe1wlGEe2JFqdYY1C
      unzip -j freqk11_permuted_samples_test_fields_lexicon_1.zip
    elif [ $FREQ_TGT -eq 11 ];
    then
      gdown https://drive.google.com/uc?id=1LeJEqT0LK9WAVgVzusbkf9vIR1ZK2xAl
      unzip -j freqk11_permuted_samples_test_lexicon_1.zip
    elif [ $FREQ_TGT -eq 20 ];
    then
      gdown https://drive.google.com/uc?id=1aaai_B0KBuY6QCZFKPrgB1-e9v84-0CU
      unzip -j freqk2_permuted_samples_test_lexicon_1.zip
    else
      gdown https://drive.google.com/uc?id=1g0viMLTUz5XNkuQHd-l1NvmiLjgVLvGq
      unzip -j permuted_samples_test_lexicon_1.zip
    fi

  elif [ $LEXICON_TGT -eq 2 ];
  then
    gdown https://drive.google.com/uc?id=1KLhkOFYGf_gFmgQ3k1ksoAdAOFmMiqWO
    unzip -j permuted_samples_test_lexicon_2.zip
  fi
fi

cd $GRAMMARS_PATH

# check valid and test files are here
if ! [[ -f "$SRC_VALID" ]]; then echo "$SRC_VALID is not found!"; exit; fi
if ! [[ -f "$TGT_VALID" ]]; then echo "$TGT_VALID is not found!"; exit; fi
if ! [[ -f "$SRC_TEST" ]]; then echo "$SRC_TEST is not found!"; exit; fi
if ! [[ -f "$TGT_TEST" ]]; then echo "$TGT_TEST is not found!"; exit; fi

echo "we reach line 510"

# tokenize data
if ! [[ -f "$SRC_VALID_TOK" && -f "$TGT_VALID_TOK" ]]; then
  echo "Tokenize Valid data..."
  cat $SRC_VALID | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_VALID_TOK
  cat $TGT_VALID | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $TGT_VALID_TOK
fi

if ! [[ -f "$SRC_TEST_TOK" && -f "$TGT_TEST_TOK" ]]; then
  echo "Tokenize Valid data..."
  cat $SRC_TEST | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TEST_TOK
  cat $TGT_TEST | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $TGT_TEST_TOK
fi

echo "we reach line 525"

#echo "Tokenizing valid and test data..."
#$INPUT_FROM_SGM < $SRC_VALID.sgm | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_VALID
#$INPUT_FROM_SGM < $TGT_VALID.sgm | $NORM_PUNC -l fr | $REM_NON_PRINT_CHAR | $TOKENIZER -l fr -no-escape -threads $N_THREADS > $TGT_VALID
#$INPUT_FROM_SGM < $SRC_TEST.sgm | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TEST
#$INPUT_FROM_SGM < $TGT_TEST.sgm | $NORM_PUNC -l fr | $REM_NON_PRINT_CHAR | $TOKENIZER -l fr -no-escape -threads $N_THREADS > $TGT_TEST

echo "Applying BPE to valid and test files..."
$FASTBPE applybpe $SRC_VALID_TOK.$CODES $SRC_VALID_TOK $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $TGT_VALID_TOK.$CODES $TGT_VALID_TOK $BPE_CODES $TGT_VOCAB
$FASTBPE applybpe $SRC_TEST_TOK.$CODES $SRC_TEST_TOK $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $TGT_TEST_TOK.$CODES $TGT_TEST_TOK $BPE_CODES $TGT_VOCAB

echo "Binarizing data..."
rm -f $SRC_VALID_TOK.$CODES.pth $TGT_VALID_TOK.$CODES.pth $SRC_TEST_TOK.$CODES.pth $TGT_TEST_TOK.$CODES.pth
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VALID_TOK.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VALID_TOK.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST_TOK.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST_TOK.$CODES


#
# Summary
#
echo ""
echo "===== Data summary"
echo "Monolingual training data:"
echo "    $SRC_NAME: $SRC_TOK.$CODES.pth"
echo "    $TGT_NAME: $TGT_TOK.$CODES.pth"
echo "Parallel validation data:"
echo "    $SRC_NAME: $SRC_VALID_TOK.$CODES.pth"
echo "    $TGT_NAME: $TGT_VALID_TOK.$CODES.pth"
echo "Parallel test data:"
echo "    $SRC_NAME: $SRC_TEST_TOK.$CODES.pth"
echo "    $TGT_NAME: $TGT_TEST_TOK.$CODES.pth"
echo ""


#
# Train fastText on concatenated embeddings
#

if ! [[ -f "$CONCAT_BPE" ]]; then
  echo "Concatenating source and target monolingual data..."
  cat $SRC_TOK.$CODES $TGT_TOK.$CODES | shuf > $CONCAT_BPE
fi
echo "Concatenated data in: $CONCAT_BPE"

if ! [[ -f "$CONCAT_BPE.vec" ]]; then
  echo "Training fastText on $CONCAT_BPE..."
  $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -thread $N_THREADS -ws 5 -neg 10 -input $CONCAT_BPE -output $CONCAT_BPE
fi
echo "Cross-lingual embeddings in: $CONCAT_BPE.vec"

# Deletin bin file because it is too big!
echo "Deleting $CONCAT_BPE.bin"
rm -r -f $CONCAT_BPE.bin
echo "Done!"
