mkdir input; cd input;

# main data
DATA=google-quest-challenge
kaggle competitions download --force -c $DATA
mkdir $DATA
unzip $DATA -d ${DATA} 

