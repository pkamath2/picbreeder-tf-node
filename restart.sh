#!/bin/bash
process_id=$(</home/ec2-user/picbreeder-tf-node/PID)
kill -9 $process_id
node /home/ec2-user/picbreeder-tf-node/picbreeder.js > /home/ec2-user/picbreeder-tf-node/picbreeder.log &
echo $! > /home/ec2-user/picbreeder-tf-node/PID
