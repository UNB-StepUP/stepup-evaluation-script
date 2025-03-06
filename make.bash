sub_folders="reference scoring example_submission"

for f in $sub_folders; do
  cd $f;
  zip -r ../$f .;
  cd ..;
done

zip -r task .