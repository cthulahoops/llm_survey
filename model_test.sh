#!/bin/bash

chatcli init
chatcli add -p mmtest --role user < prompt.md

for model in $(cat models); do
    echo "$model"
    chatcli answer --model "$model" -t ^mmtest --sync >/dev/null
done
