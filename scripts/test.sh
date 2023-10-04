nohup echo "c1"  > /dev/null 2>&1 \
&& echo "执行第二条命令" \
&& nohup  echo "c2"  > /dev/null 2>&1 \
&& echo "执行第三条命令" \
&& nohup  echo "c3"  > /dev/null 2>&1 \
&& nohup  echo "c4" > /dev/null 2>&1 &
