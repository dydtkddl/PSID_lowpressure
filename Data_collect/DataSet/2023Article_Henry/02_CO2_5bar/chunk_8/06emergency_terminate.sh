ps aux | grep "/home/yongsang/Research/simulations/bin/simulate" | grep -v "grep" | awk '{print $2}' | xargs kill -9

