# find all files with the pattern *_morpheme.json but not *_F_morpheme.json
# and delete them
find . -type f -name '*_morpheme.json' ! -name '*_F_morpheme.json' -delete