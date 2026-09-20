
for i in `ls -d -1 checkpoints-*`
do
   echo "rm -r $i"
   rm -r $i
done

for i in `git status | grep "ny fil:" | cut -c 16-200 | grep "checkpoints-"`
do
  git rm -r $i
done
