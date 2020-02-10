#!/usr/bin/env bash
set -e
set -o errexit
set -o nounset
set -o pipefail

# Change to script dir
cd "$(cd -P -- "$(dirname -- "$0")" && pwd -P)"

# Download sst2
mkdir -p SST2
cd SST2
[ -f ./dev.tsv ] ||  wget https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/dev.tsv
[ -f ./train.tsv ] || wget https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/train.tsv
[ -f ./test.tsv ] || wget https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/test.tsv

echo "Download and unzip the file inside the data folder. Press any key to continue..."
read -n1

URL="https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download"
# Download GoogleNews-vectors-300-negative
if command -v xdg-open 1> /dev/null 2>&1;then
    xdg-open "${URL}"
elif command -v open 1> /dev/null 2>&1;then
    open "${URL}"
fi
