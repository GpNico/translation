#!/bin/bash

echo "Running Lample..."

set -e

# Parameters
CODES=1000      # number of BPE codes
SRC_NAME=011111
TGT_NAME=011111
LEXICON_SRC=0
LEXICON_TGT=1
FREQ_SRC=0
FREQ_TGT=0

GR1=s$SRC_NAME.LEX$LEXICON_SRC
GR2=t$TGT_NAME.LEX$LEXICON_TGT
# Supervision
declare -i SUPERVISED
SUPERVISED=0

# Bilingual dict sup 
PARTIAL_DICT=False
BILINGUAL_DICT_SUP=False #wheter or not we do partial supervision on bilingualm dictionnary only

# Paths
DATA_PATH=$PWD/data
GRAMMARS_PATH=$DATA_PATH/artificial_grammars

SRC_STRING=$GR1
TGT_STRING=$GR2
if [ $FREQ_SRC -ge 1 ];
then
  SRC_STRING=$SRC_STRING'.FREQ_K'$FREQ_SRC
fi
if [ $FREQ_TGT -ge 1 ];
then
  TGT_STRING=$TGT_STRING'.FREQ_K'$FREQ_TGT
fi
DATA_FOLDER='GR_'$SRC_STRING'.'$TGT_STRING
# Model
TRANSFORMER=True
if [ $TRANSFORMER = True ]; 
then
  EXP_NAME=$DATA_FOLDER
else
  EXP_NAME=$DATA_FOLDER'_LSTM'
fi

if [ $SUPERVISED -eq 100 ]; 
then
  DATA_FOLDER=$DATA_FOLDER'_SUP'
fi

##########################
# Tools
  
UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools
N_THREADS=48  # number of threads in data preprocessing

FULL_VOCAB=$GRAMMARS_PATH/$DATA_FOLDER/vocab.s$SRC_NAME.LEX0-t$TGT_NAME.LEX$LEXICON_TGT.$CODES
BPE_CODES=$GRAMMARS_PATH/$DATA_FOLDER/bpe_codes$CODES.s$SRC_NAME.LEX0-t$TGT_NAME.LEX$LEXICON_TGT

SRC_VOCAB=$GRAMMARS_PATH/$DATA_FOLDER/vocab.s$SRC_NAME.LEX0.$CODES
TGT_VOCAB=$GRAMMARS_PATH/$DATA_FOLDER/vocab.t$TGT_NAME.LEX$LEXICON_TGT.$CODES

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

# Customized file
CUSTOMIZED_RAW=$GRAMMARS_PATH/customized/lexicon_0_words_only.txt
CUSTOMIZED_TOK=$GRAMMARS_PATH/customized/lexicon_0_words_only.tok

echo "test"
if [ $BILINGUAL_DICT_SUP = True ];
then
  if [ $PARTIAL_DICT = True ];
  then
    echo "Partial bilingual dict!"
    LEXICON0_RAW=$GRAMMARS_PATH/bilingual_dictionnaries/lexicon_0_words_only_partial.txt
    LEXICON0_TOK=$GRAMMARS_PATH/bilingual_dictionnaries/lexicon_0_words_only_partial.tok
    LEXICON1_RAW=$GRAMMARS_PATH/bilingual_dictionnaries/lexicon_1_words_only_partial.txt
    LEXICON1_TOK=$GRAMMARS_PATH/bilingual_dictionnaries/lexicon_1_words_only_partial.tok
  else
    echo "Full bilingual dict!"
    LEXICON0_RAW=$GRAMMARS_PATH/bilingual_dictionnaries/lexicon_0_words_only.txt
    LEXICON0_TOK=$GRAMMARS_PATH/bilingual_dictionnaries/lexicon_0_words_only.tok
    LEXICON1_RAW=$GRAMMARS_PATH/bilingual_dictionnaries/lexicon_1_words_only.txt
    LEXICON1_TOK=$GRAMMARS_PATH/bilingual_dictionnaries/lexicon_1_words_only.tok
  fi
fi

# tokenize data
echo "Tokenizing customized data..."
cat $CUSTOMIZED_RAW | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $CUSTOMIZED_TOK

  
# apply fastbpe
echo "Applying FastBPE..."
$FASTBPE applybpe $CUSTOMIZED_TOK.$CODES $CUSTOMIZED_TOK $BPE_CODES $SRC_VOCAB
  
# Binarizing data...
echo "Binarizing customized data..."
rm -f $CUSTOMIZED_TOK.$CODES.pth
$UMT_PATH/preprocess.py $FULL_VOCAB $CUSTOMIZED_TOK.$CODES

##########################


# Main code
if [ $SUPERVISED -eq 100 ];
then
  
  echo 'Supervised Training!'
  EXP_NAME=$EXP_NAME'_Supervised'
  python main.py --exp_name $EXP_NAME --batch_size 16 --wandb --max_epoch 50 --transformer $TRANSFORMER --n_enc_layers 4 --n_dec_layers 4 --share_enc 3 --share_dec 3 --share_lang_emb True --share_output_emb True --langs $GR1','$GR2 --n_para -1 --para_dataset $GR1'-'$GR2':'$GRAMMARS_PATH'/'$DATA_FOLDER'/sample_XX.tok.'$CODES'.pth,'$GRAMMARS_PATH'/'$DATA_FOLDER'/valid/sample_XX.tok.'$CODES'.pth,'$GRAMMARS_PATH'/'$DATA_FOLDER'/test/sample_XX.tok.'$CODES'.pth' --para_directions $GR1'-'$GR2','$GR2'-'$GR1 --pretrained_emb $GRAMMARS_PATH'/'$DATA_FOLDER'/all.'$GR1'-'$GR2'.'$CODES'.vec' --pretrained_out True --lambda_xe_para '0:1,100000:0.1,300000:0' --otf_num_processes 30 --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001 --epoch_size 100000 --stopping_criterion 'bleu_'$GR1'_'$GR2'_valid,20' --customized_data $GR1':'$CUSTOMIZED_TOK.$CODES.pth';'$GR2':'
  
elif [ $BILINGUAL_DICT_SUP = True ];
then
  echo 'Unsupervised training with supervised bilingual dictionnary training'
  
  # WE NEED TO DO ALL THAT BECAUSE WE NEED TO TOKENIZE WITH THE SAME BPE
  # tokenize data
  echo "Tokenizing customized data..."
  cat $LEXICON0_RAW | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $LEXICON0_TOK
  cat $LEXICON1_RAW | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $LEXICON1_TOK
    
  # apply fastbpe
  echo "Applying FastBPE..."
  $FASTBPE applybpe $LEXICON0_TOK.$CODES $LEXICON0_TOK $BPE_CODES $SRC_VOCAB
  $FASTBPE applybpe $LEXICON1_TOK.$CODES $LEXICON1_TOK $BPE_CODES $TGT_VOCAB
    
  # Binarizing data...
  echo "Binarizing customized data..."
  rm -f $LEXICON0_TOK.$CODES.pth
  rm -f $GR1.tok.$CODES.pth
  $UMT_PATH/preprocess.py $FULL_VOCAB $LEXICON0_TOK.$CODES
  rm -f $LEXICON1_TOK.$CODES.pth
  rm -f $GR2.tok.$CODES.pth
  $UMT_PATH/preprocess.py $FULL_VOCAB $LEXICON1_TOK.$CODES
  
  # Renaming to match code template but the name of the grammar is useless...
  mv $LEXICON0_TOK.$CODES.pth $GRAMMARS_PATH/bilingual_dictionnaries/$GR1.tok.$CODES.pth
  mv $LEXICON1_TOK.$CODES.pth $GRAMMARS_PATH/bilingual_dictionnaries/$GR2.tok.$CODES.pth
  
  # Call main.py
  python main.py --exp_name $EXP_NAME'_Bilingual_Dict' --batch_size 16 --wandb --max_epoch 40 --transformer $TRANSFORMER --n_enc_layers 4 --n_dec_layers 4 --share_enc 3 --share_dec 3 --share_lang_emb True --share_output_emb True --langs $GR1','$GR2 --n_mono -1 --mono_dataset $GR1':'$GRAMMARS_PATH'/'$DATA_FOLDER'/sample_'$GR1'.tok.'$CODES'.pth,,;'$GR2':'$GRAMMARS_PATH'/'$DATA_FOLDER'/sample_'$GR2'.tok.'$CODES'.pth,,' --n_para -1 --para_dataset $GR1'-'$GR2':'$GRAMMARS_PATH'/bilingual_dictionnaries/XX.tok.'$CODES'.pth,'$GRAMMARS_PATH'/'$DATA_FOLDER'/valid/sample_XX.tok.'$CODES'.pth,'$GRAMMARS_PATH'/'$DATA_FOLDER'/test/sample_XX.tok.'$CODES'.pth' --mono_directions $GR1','$GR2 --para_directions $GR1'-'$GR2','$GR2'-'$GR1 --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 --pivo_directions $GR2'-'$GR1'-'$GR2','$GR1'-'$GR2'-'$GR1 --pretrained_emb $GRAMMARS_PATH'/'$DATA_FOLDER'/all.'$GR1'-'$GR2'.'$CODES'.vec' --pretrained_out True --lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_para '0:1,100000:0.1,300000:0' --lambda_xe_otfd 1 --otf_num_processes 30 --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001 --epoch_size 100000 --stopping_criterion 'bleu_'$GR1'_'$GR2'_valid,20' --customized_data $GR1':'$CUSTOMIZED_TOK.$CODES.pth';'$GR2':'

elif [ $SUPERVISED -eq 0 ];
then

  echo 'Unsupervised Training!' 
  python main.py --exp_name $EXP_NAME --batch_size 16 --wandb --max_epoch 40 --transformer $TRANSFORMER --n_enc_layers 4 --n_dec_layers 4 --share_enc 3 --share_dec 3 --share_lang_emb True --share_output_emb True --langs $GR1','$GR2 --n_mono -1 --mono_dataset $GR1':'$GRAMMARS_PATH'/'$DATA_FOLDER'/sample_'$GR1'.tok.'$CODES'.pth,,;'$GR2':'$GRAMMARS_PATH'/'$DATA_FOLDER'/sample_'$GR2'.tok.'$CODES'.pth,,' --para_dataset $GR1'-'$GR2':,'$GRAMMARS_PATH'/'$DATA_FOLDER'/valid/sample_XX.tok.'$CODES'.pth,'$GRAMMARS_PATH'/'$DATA_FOLDER'/test/sample_XX.tok.'$CODES'.pth' --mono_directions $GR1','$GR2 --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 --pivo_directions $GR2'-'$GR1'-'$GR2','$GR1'-'$GR2'-'$GR1 --pretrained_emb $GRAMMARS_PATH'/'$DATA_FOLDER'/all.'$GR1'-'$GR2'.'$CODES'.vec' --pretrained_out True --lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_otfd 1 --otf_num_processes 30 --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001 --epoch_size 100000 --stopping_criterion 'bleu_'$GR1'_'$GR2'_valid,20' --customized_data $GR1':'$CUSTOMIZED_TOK.$CODES.pth';'$GR2':'
  
else
  echo 'Training with a proportion of '$SUPERVISED' supervised examples !'
  EXP_NAME=$EXP_NAME'_'$SUPERVISED'_Prop_Supervised'
  declare -i N_SUP_EX
  N_SUP_EX='100000*'$SUPERVISED
  N_SUP_EX=$N_SUP_EX'/100'
  echo 'N_SUP_EX '$N_SUP_EX
  
  echo "Creating target .pth..."
  # WE NEED TO DO ALL THAT BECAUSE WE NEED TO TOKENIZE WITH THE SAME BPE
  mkdir -p ./data/artificial_grammars/$DATA_FOLDER/SUP
  
  if [ $LEXICON_TGT -eq 0 ];
  then
    TGT_RAW=$GRAMMARS_PATH/source/sample_$TGT_NAME.txt
  else
    TARGET_DIR=source_lexicon$LEXICON_TGT
    TGT_RAW=$GRAMMARS_PATH/$TARGET_DIR/sample_$TGT_NAME.txt
  fi
  echo "TGT_RAW is $TGT_RAW"
  TGT_TOK=$GRAMMARS_PATH/$DATA_FOLDER/SUP/sample_t$TGT_NAME.LEX$LEXICON_TGT.tok
  
  
  # tokenize data
  if ! [[ -f "$TGT_TOK" ]]; then
    echo "Tokenizing data..."
    cat $TGT_RAW | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $TGT_TOK
  fi
  
  # apply fastbpe
  echo "Applying FastBPE..."
  $FASTBPE applybpe $TGT_TOK.$CODES $TGT_TOK $BPE_CODES $TGT_VOCAB
  
  # Binarizing data...
  echo "Binarizing data..."
  rm -f $TGT_TOK.$CODES.pth
  $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TOK.$CODES
  
  # Copy source .pth in the SUP folder
  echo "Copying source file to SUP folder..."
  cp $GRAMMARS_PATH/$DATA_FOLDER/sample_$GR1.tok.$CODES.pth $GRAMMARS_PATH/$DATA_FOLDER/SUP
  
  echo "Done! Running main.py..."
  
  # Call main.py
  python main.py --exp_name $EXP_NAME --batch_size 16 --wandb --max_epoch 40 --transformer $TRANSFORMER --n_enc_layers 4 --n_dec_layers 4 --share_enc 3 --share_dec 3 --share_lang_emb True --share_output_emb True --langs $GR1','$GR2 --n_mono -1 --mono_dataset $GR1':'$GRAMMARS_PATH'/'$DATA_FOLDER'/sample_'$GR1'.tok.'$CODES'.pth,,;'$GR2':'$GRAMMARS_PATH'/'$DATA_FOLDER'/sample_'$GR2'.tok.'$CODES'.pth,,' --n_para $N_SUP_EX --para_dataset $GR1'-'$GR2':'$GRAMMARS_PATH'/'$DATA_FOLDER'/SUP/sample_XX.tok.'$CODES'.pth,'$GRAMMARS_PATH'/'$DATA_FOLDER'/valid/sample_XX.tok.'$CODES'.pth,'$GRAMMARS_PATH'/'$DATA_FOLDER'/test/sample_XX.tok.'$CODES'.pth' --mono_directions $GR1','$GR2 --para_directions $GR1'-'$GR2','$GR2'-'$GR1 --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 --pivo_directions $GR2'-'$GR1'-'$GR2','$GR1'-'$GR2'-'$GR1 --pretrained_emb $GRAMMARS_PATH'/'$DATA_FOLDER'/all.'$GR1'-'$GR2'.'$CODES'.vec' --pretrained_out True --lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_para '0:1,100000:0.1,300000:0' --lambda_xe_otfd 1 --otf_num_processes 30 --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001 --epoch_size 100000 --stopping_criterion 'bleu_'$GR1'_'$GR2'_valid,20' --customized_data $GR1':'$CUSTOMIZED_TOK.$CODES.pth';'$GR2':'
fi
