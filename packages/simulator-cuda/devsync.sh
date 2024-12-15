#!/bin/bash

# Check if the required arguments are provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <port> <host>"
  exit 1
fi

port=$1
host=$2


sync_dir(){
  dir=$1
  rsync -azh               \
      -e "ssh -q -p $port"    \
      --delete             \
      --filter=":- ../$dir/.gitignore" \
      ../$dir $host:/root/tpaw
}

sync_dir "simulator-cuda"

# run_count=0

# run_rsync () {
#   run_count=$((run_count+1))
  # rsync -azh               \
  #     -e "ssh -q -p $port"    \
  #     --delete             \
  #     --filter=':- .gitignore' \
  #     . $host:/root/tpaw/simulator-rust
#   echo "Sync $run_count"
# }

# run_rsync

# fswatch -o . | while read -r file; do
#   run_rsync
# done