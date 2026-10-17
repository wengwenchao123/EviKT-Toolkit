
dataset="ednet"
models="dkt,akt,atdkt,dkvmn,ukt,keenkt,simplekt,robustkt"

IFS=','
for i in $models; do
  echo "# $i"
  python -u extract_raw_result.py --dataset $dataset --model_name $i
  echo ""
done


# nohup sh extract_raw.sh >extract.log&