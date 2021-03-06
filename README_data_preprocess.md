# Download Data from GEO

1. Find PHASE 1 LINCS [dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)
2. Reference PHASE 1 LINCS [user guide](https://docs.google.com/document/d/1q2gciWRhVCAAnlvF2iRLuJ7whrGP6QjpsCMq1yWz7dU/edit#heading=h.usef9o7fuux3)
3. Download the Level3 expression data

```
URLBASE='ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/'
DATADIR='mnt/knowdnn_hdd/tfuser/data/from_GEO/'
wget -P $DATADIR $URLBASE/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz
gunzip $DATADIR/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz
```

4. Download the experiment metadata

```
wget -P $DATADIR/lincs_meta $URLBASE/GSE92742_Broad_LINCS_inst_info.txt.gz
gunzip $DATADIR/lincs_meta/GSE92742_Broad_LINCS_inst_info.txt.gz
```

5. Download the probe metadata

```
wget -P $DATADIR/lincs_meta $URLBASE/GSE92742_Broad_LINCS_gene_info.txt.gz
gunzip $DATADIR/lincs_meta/GSE92742_Broad_LINCS_gene_info.txt.gz

```

7. Extract only the L1000 probes into GSE92742_Broad_LINCS_gene_info.l1000.txt
```
cut -f2 GSE92742_Broad_LINCS_gene_info.l1000.txt > GSE92742_Broad_LINCS_gene_info.l1000.genes.txt
docker run --rm -w /mnt/knowdnn_hdd/tfuser/data/from_GEO/lincs_meta -v /mnt/knowdnn_hdd/tfuser/data/from_GEO/lincs_meta:/mnt/knowdnn_hdd/tfuser/data/from_GEO/lincs_meta knoweng/kn_mapper /home/src/kn_mapper.py GSE92742_Broad_LINCS_gene_info.l1000.genes.txt -t 9606
```

8. Download the other lincs metadata

9. Download the mcf10a experiment data
```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/mcf10a_progression/'
mkdir -p $DATADIR
wget -P $DATADIR https://media.nature.com/original/nature-assets/ncomms/2015/150309/ncomms7367/extref/ncomms7367-s4.xlsx
# save table as file original_log2RPKM_table.txt
cut -f1 original_log2RPKM_table.txt > original_log2RPKM_table.genes.txt
docker run --rm -w $DATADIR -v $DATADIR:$DATADIR knoweng/kn_mapper /home/src/kn_mapper.py  original_log2RPKM_table.genes.txt -t 9606
```

# Create Filtered Metadata
## Setup Knockdown Filtered
1. Summarize all LINCS metadata:
```
INDIR='/mnt/knowdnn_hdd/tfuser/data/from_GEO/lincs_meta'
OUTDIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/all_exps'
CODEDIR='/home/tfuser/code/preprocess/'
mkdir -p $OUTDIR/
python3 $CODEDIR/summarize_metadata.py $INDIR/GSE92742_Broad_LINCS_inst_info.txt \
    $OUTDIR/all_exps.pert_inames.txt > $OUTDIR/all_exps.summary.log
```

2. Filter only knockdown LINCS metadata:

```
INDIR='/mnt/knowdnn_hdd/tfuser/data/from_GEO/lincs_meta'
OUTDIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh'
CODEDIR='/home/tfuser/code/preprocess/'
mkdir -p $OUTDIR/
python3 $CODEDIR/filter_metadata.py $INDIR/GSE92742_Broad_LINCS_inst_info.txt \
    $OUTDIR/trt_sh.metadata.txt -pt trt_sh
python3 $CODEDIR/summarize_metadata.py $OUTDIR/trt_sh.metadata.txt \
    $OUTDIR/trt_sh.pert_inames.txt > $OUTDIR/trt_sh.summary.log
```

  - 448737 trt_sh exps, at least 21 per 20 cell_ids, at least 6 per 4369 pert_inames

## top50 Knockdown Filtered
1. Save top 50 knockdown classes to file:

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh'
head -n 50 $DATADIR/trt_sh.pert_inames.txt > $DATADIR/trt_sh.pert_inames.top50.txt
```

2. Select top 50 knockdown classes and keep 10 largest cell lines (ASC is 11th):

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh'
CODEDIR='/home/tfuser/code/preprocess/'
OUTDIR=$DATADIR/top50/
mkdir -p $OUTDIR

OUTTAG='top50'
OUTMETA=$OUTDIR/$OUTTAG.metadata.txt
python3 $CODEDIR/filter_metadata.py $DATADIR/trt_sh.metadata.txt $OUTMETA \
    -ci VCAP,PC3,MCF7,HT29,A375,HA1E,A549,HEPG2,HCC515,NPC \
    -pif $DATADIR/trt_sh.pert_inames.top50.txt > $OUTDIR/$OUTTAG.metadata.log
python3 $CODEDIR/summarize_metadata.py $OUTMETA \
    $OUTDIR/$OUTTAG.pert_inames.txt > $OUTDIR/$OUTTAG.summary.log
```

  - 30010 trt_sh exps, at least 780 per 10 cell_ids, at least 375 per 50 pert_inames

3. Select top 50 knockdown classes and keep ASC (11th largest) cell_id:

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh'
CODEDIR='/home/tfuser/code/preprocess/'
OUTDIR=$DATADIR/top50/
mkdir -p $OUTDIR

OUTTAG='top50.ASC'
OUTMETA=$OUTDIR/$OUTTAG.metadata.txt
python3 $CODEDIR/filter_metadata.py $DATADIR/trt_sh.metadata.txt $OUTMETA \
    -ci ASC \
    -pif $DATADIR/trt_sh.pert_inames.top50.txt > $OUTDIR/$OUTTAG.metadata.log
python3 $CODEDIR/summarize_metadata.py $OUTMETA \
    $OUTDIR/$OUTTAG.pert_inames.txt > $OUTDIR/$OUTTAG.summary.log
```

  - 1638 trt_sh exps, at least 93 per 7 cell_ids, at least 6 per 32 pert_inames

4. Select top 50 knockdown classes and keep small test cell lines:

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh'
CODEDIR='/home/tfuser/code/preprocess/'
OUTDIR=$DATADIR/top50/
mkdir -p $OUTDIR

OUTTAG='top50.hiddenCL'
OUTMETA=$OUTDIR/$OUTTAG.metadata.txt
python3 $CODEDIR/filter_metadata.py $DATADIR/trt_sh.metadata.txt $OUTMETA \
    -ci ASC,U2OS,HEK293T,HEKTE,SW480,SKL,MCH58 \
    -pif $DATADIR/trt_sh.pert_inames.top50.txt > $OUTDIR/$OUTTAG.metadata.log
python3 $CODEDIR/summarize_metadata.py $OUTMETA \
    $OUTDIR/$OUTTAG.pert_inames.txt > $OUTDIR/$OUTTAG.summary.log
```

  - 674 trt_sh exps, at least 674 per 1 cell_ids, at least 6 per 43 pert_inames

## Bottom 4000 Knockdown Filtered
1. Save bottom 4000 knockdown classes to file (starting from 401st):

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh'
tail -n+401 $DATADIR/trt_sh.pert_inames.txt > $DATADIR/trt_sh.pert_inames.b4k.txt
# manually edit renumber class labels and drop classes < 10
```

2. Select bottom 4000 knockdown classes and keep 10 largest cell lines (ASC is 11th):

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh'
CODEDIR='/home/tfuser/code/preprocess/'
OUTDIR=$DATADIR/b4k/
mkdir -p $OUTDIR

OUTTAG='b4k'
OUTMETA=$OUTDIR/$OUTTAG.metadata.txt
python3 $CODEDIR/filter_metadata.py $DATADIR/trt_sh.metadata.txt $OUTMETA \
    -ci VCAP,PC3,MCF7,HT29,A375,HA1E,A549,HEPG2,HCC515,NPC \
    -pif $DATADIR/trt_sh.pert_inames.b4k.txt > $OUTDIR/$OUTTAG.metadata.log
python3 $CODEDIR/summarize_metadata.py $OUTMETA \
    $OUTDIR/$OUTTAG.pert_inames.txt > $OUTDIR/$OUTTAG.summary.log
```

  - 311631 trt_sh exps, at least 8353 per 10 cell_ids, at least 5 per 3959 pert_inames

3. Select bottom 4000 knockdown classes and keep ASC (11th largest) cell_id:

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh'
CODEDIR='/home/tfuser/code/preprocess/'
OUTDIR=$DATADIR/b4k/
mkdir -p $OUTDIR

OUTTAG='b4k.ASC'
OUTMETA=$OUTDIR/$OUTTAG.metadata.txt
python3 $CODEDIR/filter_metadata.py $DATADIR/trt_sh.metadata.txt $OUTMETA \
    -ci ASC \
    -pif $DATADIR/trt_sh.pert_inames.b4k.txt > $OUTDIR/$OUTTAG.metadata.log
python3 $CODEDIR/summarize_metadata.py $OUTMETA \
    $OUTDIR/$OUTTAG.pert_inames.txt > $OUTDIR/$OUTTAG.summary.log
```

  - 7943 trt_sh exps, at least 7943 per 1 cell_ids, at least 4 per 799 pert_inames

4. Select bottom 4000 knockdown classes and keep small hidden cell lines:

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh'
CODEDIR='/home/tfuser/code/preprocess/'
OUTDIR=$DATADIR/b4k/
mkdir -p $OUTDIR

OUTTAG='b4k.hiddenCL'
OUTMETA=$OUTDIR/$OUTTAG.metadata.txt
python3 $CODEDIR/filter_metadata.py $DATADIR/trt_sh.metadata.txt $OUTMETA \
    -ci ASC,U2OS,HEK293T,HEKTE,SW480,SKL,MCH58 \
    -pif $DATADIR/trt_sh.pert_inames.b4k.txt > $OUTDIR/$OUTTAG.metadata.log
python3 $CODEDIR/summarize_metadata.py $OUTMETA \
    $OUTDIR/$OUTTAG.pert_inames.txt > $OUTDIR/$OUTTAG.summary.log
```

  - 15835 trt_sh exps, at least 71 per 7 cell_ids, at least 4 per 1111 pert_inames


## top 400 Knockdown Filtered
1. Save top 400 knockdown classes to file:

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh'
head -n 400 $DATADIR/trt_sh.pert_inames.txt > $DATADIR/trt_sh.pert_inames.top400.txt
```

2. Select top 400 knockdown classes and keep 10 largest cell lines (ASC is 11th):

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh'
CODEDIR='/home/tfuser/code/preprocess/'
OUTDIR=$DATADIR/top400/
mkdir -p $OUTDIR

OUTTAG='top400'
OUTMETA=$OUTDIR/$OUTTAG.metadata.txt
python3 $CODEDIR/filter_metadata.py $DATADIR/trt_sh.metadata.txt $OUTMETA \
    -ci VCAP,PC3,MCF7,HT29,A375,HA1E,A549,HEPG2,HCC515,NPC \
    -pif $DATADIR/trt_sh.pert_inames.top400.txt > $OUTDIR/$OUTTAG.metadata.log
python3 $CODEDIR/summarize_metadata.py $OUTMETA \
    $OUTDIR/$OUTTAG.pert_inames.txt > $OUTDIR/$OUTTAG.summary.log
```

  - 112800 trt_sh exps, at least 3109 per 10 cell_ids, at least 174 per 400 pert_inames

3. Select top 400 knockdown classes and keep ASC (11th largest) cell_id:

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh'
CODEDIR='/home/tfuser/code/preprocess/'
OUTDIR=$DATADIR/top400/
mkdir -p $OUTDIR

OUTTAG='top400.ASC'
OUTMETA=$OUTDIR/$OUTTAG.metadata.txt
python3 $CODEDIR/filter_metadata.py $DATADIR/trt_sh.metadata.txt $OUTMETA \
    -ci ASC \
    -pif $DATADIR/trt_sh.pert_inames.top400.txt > $OUTDIR/$OUTTAG.metadata.log
python3 $CODEDIR/summarize_metadata.py $OUTMETA \
    $OUTDIR/$OUTTAG.pert_inames.txt > $OUTDIR/$OUTTAG.summary.log
```

  - 3155 trt_sh exps, at least 3155 per 1 cell_ids, at least 4 per 275 pert_inames

4. Select top 400 knockdown classes and keep small hidden cell lines:

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh'
CODEDIR='/home/tfuser/code/preprocess/'
OUTDIR=$DATADIR/top400/
mkdir -p $OUTDIR

OUTTAG='top400.hiddenCL'
OUTMETA=$OUTDIR/$OUTTAG.metadata.txt
python3 $CODEDIR/filter_metadata.py $DATADIR/trt_sh.metadata.txt $OUTMETA \
    -ci ASC,U2OS,HEK293T,HEKTE,SW480,SKL,MCH58 \
    -pif $DATADIR/trt_sh.pert_inames.top400.txt > $OUTDIR/$OUTTAG.metadata.log
python3 $CODEDIR/summarize_metadata.py $OUTMETA \
    $OUTDIR/$OUTTAG.pert_inames.txt > $OUTDIR/$OUTTAG.summary.log
```

  - 6693 trt_sh exps, at least 141 per 7 cell_ids, at least 4 per 333 pert_inames


## Control Experiments Filtered
1. Filter only control LINCS metadata:

```
INDIR='/mnt/knowdnn_hdd/tfuser/data/from_GEO/lincs_meta'
OUTDIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/ctl'
CODEDIR='/home/tfuser/code/preprocess/'
mkdir -p $OUTDIR/
python3 $CODEDIR/filter_metadata.py $INDIR/GSE92742_Broad_LINCS_inst_info.txt \
    $OUTDIR/ctl.metadata.txt -pt ctl_vector,ctl_vehicle,ctl_untrt
python3 $CODEDIR/summarize_metadata.py $OUTDIR/ctl.metadata.txt \
    $OUTDIR/ctl.pert_inames.txt > $OUTDIR/ctl.summary.log
```

  - 80935 ctl exps, at least 9 per 76 cell_ids, at least 2 per 15 pert_inames

2. Select ctls for 10 largest cell lines (ASC is 11th):

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/ctl'
CODEDIR='/home/tfuser/code/preprocess/'
OUTTAG="ctl_cl10"
python3 $CODEDIR/filter_metadata.py $DATADIR/ctl.metadata.txt \
    $DATADIR/$OUTTAG.metadata.txt \
    -ci VCAP,PC3,MCF7,HT29,A375,HA1E,A549,HEPG2,HCC515,NPC
python3 $CODEDIR/summarize_metadata.py $DATADIR/$OUTTAG.metadata.txt \
    $DATADIR/$OUTTAG.pert_inames.txt > $DATADIR/$OUTTAG.summary.log
```

  - 72189 ctl exps, at least 2242 per 10 cell_ids, at least 88 per 13 pert_inames

3. Select ctls for ASC (11th largest) cell_id:

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/ctl'
CODEDIR='/home/tfuser/code/preprocess/'
OUTTAG="ctl.ASC"
python3 $CODEDIR/filter_metadata.py $DATADIR/ctl.metadata.txt \
    $DATADIR/$OUTTAG.metadata.txt -ci ASC
python3 $CODEDIR/summarize_metadata.py $DATADIR/$OUTTAG.metadata.txt \
    $DATADIR/$OUTTAG.pert_inames.txt > $DATADIR/$OUTTAG.summary.log
```

  - 1838 ctl exps, at least 1838 per 1 cell_ids, at least 120 per 8 pert_inames

4. Select ctls for small hidden cell lines:

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/ctl'
CODEDIR='/home/tfuser/code/preprocess/'
OUTTAG="ctl.hiddenCL"
python3 $CODEDIR/filter_metadata.py $DATADIR/ctl.metadata.txt \
    $DATADIR/$OUTTAG.metadata.txt -ci ASC,U2OS,HEK293T,HEKTE,SW480,SKL,MCH58
python3 $CODEDIR/summarize_metadata.py $DATADIR/$OUTTAG.metadata.txt \
    $DATADIR/$OUTTAG.pert_inames.txt > $DATADIR/$OUTTAG.summary.log
```

  - 3951 ctl exps, at least 113 per 7 cell_ids, at least 7 per 9 pert_inames


# Partition Training, dev/validation, and Testing Metadata

1. Separate top50 metadata into 80,5,15 train,dev,test datasets

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh/top50'
CODEDIR='/home/tfuser/code/preprocess/'
OUTTAG='top50'
python3 $CODEDIR/partition_metadata.py $DATADIR/$OUTTAG.metadata.txt \
    $DATADIR/$OUTTAG. -pk pert_iname -pp 80,5,15 -pn train80,dev05,test15
for i in train80 dev05 test15; do
    python3 $CODEDIR/summarize_metadata.py $DATADIR/$OUTTAG.$i.metadata.txt \
        $DATADIR/$OUTTAG.$i.pert_inames.txt > $DATADIR/$OUTTAG.$i.summary.log
done;
```

  - min per pert_iname: train-300 dev-18 test-57

2. Separate bottom 4000 metadata into 80,5,15 train,dev,test datasets

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh/b4k'
CODEDIR='/home/tfuser/code/preprocess/'
OUTTAG='b4k'
python3 $CODEDIR/partition_metadata.py $DATADIR/$OUTTAG.metadata.txt \
    $DATADIR/$OUTTAG. -pk pert_iname -pp 80,5,15 -pn train80,dev05,test15
for i in train80 dev05 test15; do
    python3 $CODEDIR/summarize_metadata.py $DATADIR/$OUTTAG.$i.metadata.txt \
        $DATADIR/$OUTTAG.$i.pert_inames.txt > $DATADIR/$OUTTAG.$i.summary.log
done;
```

  - min per pert_iname: train-4 dev-1 test-1

3. Separate top400 metadata into 80,5,15 train,dev,test datasets

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/trt_sh/top400'
CODEDIR='/home/tfuser/code/preprocess/'
OUTTAG='top400'
python3 $CODEDIR/partition_metadata.py $DATADIR/$OUTTAG.metadata.txt \
    $DATADIR/$OUTTAG. -pk pert_iname -pp 80,5,15 -pn train80,dev05,test15
for i in train80 dev05 test15; do
    python3 $CODEDIR/summarize_metadata.py $DATADIR/$OUTTAG.$i.metadata.txt \
        $DATADIR/$OUTTAG.$i.pert_inames.txt > $DATADIR/$OUTTAG.$i.summary.log
done;
```

  - min per pert_iname: train-139 dev-8 test-27

4. Separate control 10 cell line metadata into 80,5,15 train,dev,test datasets

```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/ctl'
CODEDIR='/home/tfuser/code/preprocess/'
OUTTAG='ctl_cl10'
python3 $CODEDIR/partition_metadata.py $DATADIR/$OUTTAG.metadata.txt \
    $DATADIR/$OUTTAG. -pk cell_id -pp 80,5,15 -pn train80,dev05,test15
for i in train80 dev05 test15; do
    python3 $CODEDIR/summarize_metadata.py $DATADIR/$OUTTAG.$i.metadata.txt \
        $DATADIR/$OUTTAG.$i.pert_inames.txt > $DATADIR/$OUTTAG.$i.summary.log
done;
```

  - min per cell_id: train-1793 dev-112 test-337

# Create Matched Pairs

## Matched Pairs for top50
```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/'
OUTDIR='/mnt/knowdnn_hdd/tfuser/data/paired_metadata/'
CODEDIR='/home/tfuser/code/preprocess'

for MATCHSTR in \
top50:ctl/ctl_cl10:trt_sh/top50/top50:2::dev05:100 \
top50:ctl/ctl_cl10:trt_sh/top50/top50:2::train80:1000 \
top50:ctl/ctl:trt_sh/top50/top50:2::ASC:10 \
top50:ctl/ctl_cl10:trt_sh/top50/top50:2::test15:300 \
top50:ctl/ctl_cl10:trt_sh/top50/top50:2:-um:test15:300 \
top50:trt_sh/top50/top50:ctl/ctl_cl10:1::test15:300 \
top50:trt_sh/top50/top50:trt_sh/top50/top50:2::test15:300 \
top50:trt_sh/top50/top50:trt_sh/top50/top50:2:-um:test15:300 \
; do

    COLL=`echo $MATCHSTR | cut -f1 -d:`
    F1STR=`echo $MATCHSTR | cut -f2 -d:`
    F2STR=`echo $MATCHSTR | cut -f3 -d:`
    CLASS_STATE=`echo $MATCHSTR | cut -f4 -d:`
    MATCH=`echo $MATCHSTR | cut -f5 -d:`
    PARTITION=`echo $MATCHSTR | cut -f6 -d:`
    NPC=`echo $MATCHSTR | cut -f7 -d:`

    S1FILE=$DATADIR/$F1STR.$PARTITION.metadata.txt
    S2FILE=$DATADIR/$F2STR.$PARTITION.metadata.txt
    CLASS_LABEL_FILE=$DATADIR/trt_sh/trt_sh.pert_inames.$COLL.txt
    KEY1=`echo $F1STR | sed 's#ctl/##g' | sed 's#trt_sh/##g' | sed "s#$COLL/##g"`
    KEY2=`echo $F2STR | sed 's#ctl/##g' | sed 's#trt_sh/##g' | sed "s#$COLL/##g"`
    mkdir -p $OUTDIR/$COLL/
    OUTFILE=$OUTDIR/$COLL/$PARTITION.$KEY1.$KEY2$MATCH.txt
    LOGFILE=$OUTDIR/$COLL/$PARTITION.$KEY1.$KEY2$MATCH.log

    CMD="python3 $CODEDIR/pair_metadata.py $S1FILE $S2FILE $OUTFILE \
        -clf $CLASS_LABEL_FILE -cs $CLASS_STATE $MATCH -npc $NPC >> $LOGFILE"
    echo $CMD
    echo $CMD > $LOGFILE
    eval $CMD
done
```

## Matched Pairs for 51st
```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/'
OUTDIR='/mnt/knowdnn_hdd/tfuser/data/paired_metadata/'
CODEDIR='/home/tfuser/code/preprocess'

for MATCHSTR in \
b4k:ctl/ctl_cl10:trt_sh/b4k/b4k:2::dev05:1 \
b4k:ctl/ctl_cl10:trt_sh/b4k/b4k:2::train80:5 \
b4k:ctl/ctl:trt_sh/b4k/b4k:2::ASC:1 \
b4k:ctl/ctl_cl10:trt_sh/b4k/b4k:2::test15:3 \
b4k:ctl/ctl_cl10:trt_sh/b4k/b4k:2:-um:test15:3 \
b4k:trt_sh/b4k/b4k:ctl/ctl_cl10:1::test15:3 \
b4k:trt_sh/b4k/b4k:trt_sh/b4k/b4k:2::test15:3 \
b4k:trt_sh/b4k/b4k:trt_sh/b4k/b4k:2:-um:test15:3 \
; do

    COLL=`echo $MATCHSTR | cut -f1 -d:`
    F1STR=`echo $MATCHSTR | cut -f2 -d:`
    F2STR=`echo $MATCHSTR | cut -f3 -d:`
    CLASS_STATE=`echo $MATCHSTR | cut -f4 -d:`
    MATCH=`echo $MATCHSTR | cut -f5 -d:`
    PARTITION=`echo $MATCHSTR | cut -f6 -d:`
    NPC=`echo $MATCHSTR | cut -f7 -d:`

    S1FILE=$DATADIR/$F1STR.$PARTITION.metadata.txt
    S2FILE=$DATADIR/$F2STR.$PARTITION.metadata.txt
    CLASS_LABEL_FILE=$DATADIR/trt_sh/trt_sh.pert_inames.$COLL.txt
    KEY1=`echo $F1STR | sed 's#ctl/##g' | sed 's#trt_sh/##g' | sed "s#$COLL/##g"`
    KEY2=`echo $F2STR | sed 's#ctl/##g' | sed 's#trt_sh/##g' | sed "s#$COLL/##g"`
    mkdir -p $OUTDIR/$COLL/
    OUTFILE=$OUTDIR/$COLL/$PARTITION.$KEY1.$KEY2$MATCH.txt
    LOGFILE=$OUTDIR/$COLL/$PARTITION.$KEY1.$KEY2$MATCH.log

    CMD="python3 $CODEDIR/pair_metadata.py $S1FILE $S2FILE $OUTFILE \
        -clf $CLASS_LABEL_FILE -cs $CLASS_STATE $MATCH -npc $NPC >> $LOGFILE"
    echo $CMD
    echo $CMD > $LOGFILE
    eval $CMD
done
```

## Merge Matched Pairs for top 51
```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/paired_metadata/'
CODEDIR='/home/tfuser/code/preprocess'

REALKEY='top50'
NONEKEY='b4k'
OUTKEY='top51'
OUTDIR="/mnt/knowdnn_hdd/tfuser/data/paired_metadata/$OUTKEY"
mkdir -p $OUTDIR

for FILE1 in `ls $DATADIR/$REALKEY/*txt`; do
    echo $FILE1
    FILE2=`echo $FILE1 | sed "s#$REALKEY#$NONEKEY#g"`
    OUTFILE=`echo $FILE1 | sed "s#$REALKEY#$OUTKEY#g"`
    LOGFILE=`echo $OUTFILE | sed "s#.txt#.log#g"`
    echo $FILE2
    echo $OUTFILE
    ls -l $FILE2
    CMD="python3 $CODEDIR/merge_paired_metadata.py $FILE1 $FILE2 $OUTFILE \
        -sl2 50 >> $LOGFILE"
    echo $CMD
    echo $CMD > $LOGFILE
    eval $CMD
done
# mv /mnt/knowdnn_hdd/tfuser/data/paired_metadata/b4k /mnt/knowdnn_hdd/tfuser/data/paired_metadata/b4k-51
```

## Matched Pairs for top400
```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/'
OUTDIR='/mnt/knowdnn_hdd/tfuser/data/paired_metadata/'
CODEDIR='/home/tfuser/code/preprocess'

COLL="top400"
for MATCHSTR in \
ctl/ctl_cl10:trt_sh/$COLL/$COLL:2::dev05:100 \
ctl/ctl_cl10:trt_sh/$COLL/$COLL:2::train80:10000 \
ctl/ctl:trt_sh/$COLL/$COLL:2::ASC:20 \
ctl/ctl_cl10:trt_sh/$COLL/$COLL:2::test15:1000 \
ctl/ctl_cl10:trt_sh/$COLL/$COLL:2:-um:test15:1000 \
trt_sh/$COLL/$COLL:ctl/ctl_cl10:1::test15:1000 \
trt_sh/$COLL/$COLL:trt_sh/$COLL/$COLL:2::test15:1000 \
trt_sh/$COLL/$COLL:trt_sh/$COLL/$COLL:2:-um:test15:1000 \
; do

    F1STR=`echo $MATCHSTR | cut -f1 -d:`
    F2STR=`echo $MATCHSTR | cut -f2 -d:`
    CLASS_STATE=`echo $MATCHSTR | cut -f3 -d:`
    MATCH=`echo $MATCHSTR | cut -f4 -d:`
    PARTITION=`echo $MATCHSTR | cut -f5 -d:`
    NPC=`echo $MATCHSTR | cut -f6 -d:`

    S1FILE=$DATADIR/$F1STR.$PARTITION.metadata.txt
    S2FILE=$DATADIR/$F2STR.$PARTITION.metadata.txt
    CLASS_LABEL_FILE=$DATADIR/trt_sh/trt_sh.pert_inames.$COLL.txt
    KEY1=`echo $F1STR | sed 's#ctl/##g' | sed 's#trt_sh/##g' | sed "s#$COLL/##g"`
    KEY2=`echo $F2STR | sed 's#ctl/##g' | sed 's#trt_sh/##g' | sed "s#$COLL/##g"`
    mkdir -p $OUTDIR/$COLL/
    OUTFILE=$OUTDIR/$COLL/$PARTITION.$KEY1.$KEY2$MATCH.txt
    LOGFILE=$OUTDIR/$COLL/$PARTITION.$KEY1.$KEY2$MATCH.log

    CMD="python3 $CODEDIR/pair_metadata.py $S1FILE $S2FILE $OUTFILE \
        -clf $CLASS_LABEL_FILE -cs $CLASS_STATE $MATCH -npc $NPC >> $LOGFILE"
    echo $CMD
    echo $CMD > $LOGFILE
    eval $CMD
done
```

## Matched Pairs for 401st
```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/filtered_metadata/'
OUTDIR='/mnt/knowdnn_hdd/tfuser/data/paired_metadata/'
CODEDIR='/home/tfuser/code/preprocess'

COLL="b4k"
for MATCHSTR in \
ctl/ctl_cl10:trt_sh/$COLL/$COLL:2::dev05:5 \
ctl/ctl_cl10:trt_sh/$COLL/$COLL:2::train80:500 \
ctl/ctl:trt_sh/$COLL/$COLL:2::ASC:1 \
ctl/ctl_cl10:trt_sh/$COLL/$COLL:2::test15:50 \
ctl/ctl_cl10:trt_sh/$COLL/$COLL:2:-um:test15:50 \
trt_sh/$COLL/$COLL:ctl/ctl_cl10:1::test15:50 \
trt_sh/$COLL/$COLL:trt_sh/$COLL/$COLL:2::test15:50 \
trt_sh/$COLL/$COLL:trt_sh/$COLL/$COLL:2:-um:test15:50 \
; do

    F1STR=`echo $MATCHSTR | cut -f1 -d:`
    F2STR=`echo $MATCHSTR | cut -f2 -d:`
    CLASS_STATE=`echo $MATCHSTR | cut -f3 -d:`
    MATCH=`echo $MATCHSTR | cut -f4 -d:`
    PARTITION=`echo $MATCHSTR | cut -f5 -d:`
    NPC=`echo $MATCHSTR | cut -f6 -d:`

    S1FILE=$DATADIR/$F1STR.$PARTITION.metadata.txt
    S2FILE=$DATADIR/$F2STR.$PARTITION.metadata.txt
    CLASS_LABEL_FILE=$DATADIR/trt_sh/trt_sh.pert_inames.$COLL.txt
    KEY1=`echo $F1STR | sed 's#ctl/##g' | sed 's#trt_sh/##g' | sed "s#$COLL/##g"`
    KEY2=`echo $F2STR | sed 's#ctl/##g' | sed 's#trt_sh/##g' | sed "s#$COLL/##g"`
    mkdir -p $OUTDIR/$COLL/
    OUTFILE=$OUTDIR/$COLL/$PARTITION.$KEY1.$KEY2$MATCH.txt
    LOGFILE=$OUTDIR/$COLL/$PARTITION.$KEY1.$KEY2$MATCH.log

    CMD="python3 $CODEDIR/pair_metadata.py $S1FILE $S2FILE $OUTFILE \
        -clf $CLASS_LABEL_FILE -cs $CLASS_STATE $MATCH -npc $NPC >> $LOGFILE"
    echo $CMD
    echo $CMD > $LOGFILE
    eval $CMD
done
```

## Merge Matched Pairs for top 401
```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/paired_metadata/'
CODEDIR='/home/tfuser/code/preprocess'

REALKEY='top400'
NONEKEY='b4k'
OUTKEY='top401'
OUTDIR="/mnt/knowdnn_hdd/tfuser/data/paired_metadata/$OUTKEY"
mkdir -p $OUTDIR

for FILE1 in `ls $DATADIR/$REALKEY/*txt`; do
    echo $FILE1
    FILE2=`echo $FILE1 | sed "s#$REALKEY#$NONEKEY#g"`
    OUTFILE=`echo $FILE1 | sed "s#$REALKEY#$OUTKEY#g"`
    LOGFILE=`echo $OUTFILE | sed "s#.txt#.log#g"`
    echo $FILE2
    echo $OUTFILE
    ls -l $FILE2
    CMD="python3 $CODEDIR/merge_paired_metadata.py $FILE1 $FILE2 $OUTFILE \
        -sl2 400 >> $LOGFILE"
    echo $CMD
    echo $CMD > $LOGFILE
    eval $CMD
done
# mv /mnt/knowdnn_hdd/tfuser/data/paired_metadata/b4k /mnt/knowdnn_hdd/tfuser/data/paired_metadata/b4k-401
```

# Extract GEO Expression Values
## Extract data for Top 50, Top 51
```
# need python2, cmappy, yaml - source activate python2 && pip install cmapPy
DATADIR='/mnt/knowdnn_hdd/tfuser/data/paired_metadata/'
CODEDIR='/home/tfuser/code/preprocess'
GEODIR="/mnt/knowdnn_hdd/tfuser/data/from_GEO/"
#GCTXFILE="/home/tfuser/data/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.top401.gctx"
GCTXFILE="/home/tfuser/data/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx"
PROBEFILE="$GEODIR/lincs_meta/GSE92742_Broad_LINCS_gene_info.l1000.txt"
OUTDIR="/mnt/knowdnn_hdd/tfuser/data/expression_examples/"

#for COLL in top50 top51; do
#    for PAIRFILE in `ls $DATADIR/$COLL/*txt`; do
for COLL in top50 ; do
    for PAIRFILE in `ls $DATADIR/$COLL/hidden*txt`; do
        KEY=`echo $PAIRFILE | sed "s#$DATADIR/$COLL/##g" | sed "s#.txt##g"`
        echo $KEY $FILE
        PAIROUTDIR="$OUTDIR/$COLL/$KEY"
        mkdir -p $PAIROUTDIR
        LOGFILE="$PAIROUTDIR/$KEY.log"
        CMD="python $CODEDIR/extract_features.py $PAIRFILE $GCTXFILE $PROBEFILE $PAIROUTDIR \
            -mc 1 >> $LOGFILE 2> $LOGFILE.warn"
        echo $CMD
        echo $CMD > $LOGFILE
        eval time $CMD
    done;
done
```

## Extract for Top400, Top401
```
# need python2, cmappy, yaml - source activate python2 && pip install cmapPy
DATADIR='/mnt/knowdnn_hdd/tfuser/data/paired_metadata/'
CODEDIR='/home/tfuser/code/preprocess'
GEODIR="/mnt/knowdnn_hdd/tfuser/data/from_GEO/"
GCTXFILE="/home/tfuser/data/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.top401.gctx"
PROBEFILE="$GEODIR/lincs_meta/GSE92742_Broad_LINCS_gene_info.l1000.txt"
OUTDIR="/mnt/knowdnn_hdd/tfuser/data/expression_examples/"

for COLL in top401; do
    for TYPEMAX in ASC:1 dev:1 test:10 train:40 ; do
        TYPE=`echo $TYPEMAX | cut -f1 -d:`
        MAX=`echo $TYPEMAX | cut -f2 -d:`
        for PAIRFILE in `ls $DATADIR/$COLL/$TYPE*txt`; do
            KEY=`echo $PAIRFILE | sed "s#$DATADIR/$COLL/##g" | sed "s#.txt##g"`
            echo $KEY $FILE
            PAIROUTDIR="$OUTDIR/$COLL/$KEY"
            mkdir -p $PAIROUTDIR
            LOGFILE="$PAIROUTDIR/$KEY.log"
            CMD="python $CODEDIR/extract_features.py $PAIRFILE $GCTXFILE $PROBEFILE $PAIROUTDIR \
                -mc $MAX >> $LOGFILE 2> $LOGFILE.warn"
            echo $CMD
            echo $CMD > $LOGFILE
            eval time $CMD
        done;
    done;
done
```

## Create MCF10A and TCGA dataset
```
OUTDIR='/mnt/knowdnn_hdd/tfuser/data/expression_examples/mcf10a_progression_znorm/'
mkdir -p  $OUTDIR
# copy from final tab of mcf10a-original-data.xlsx into $OUTDIR/file0.data
# save info.yml
echo "class_column: 1957
num_examples: 144
num_features: 1956
num_files: 1
num_metadata: 4" > $OUTDIR/info.yml

OUTDIR='/mnt/knowdnn_hdd/tfuser/data/expression_examples/tcga_paired_znorm/'
mkdir -p  $OUTDIR
# copy from final tab of cgc_tcga_pairs.xlsx into $OUTDIR/file0.data
# save info.yml
echo "class_column: 1957
num_examples: 900
num_features: 1956
num_files: 1
num_metadata: 13" > $OUTDIR/info.yml
```

# Serialize Features Files
## Serialize for Top 50 51 400 401
```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/expression_examples/'
CODEDIR='/home/tfuser/code/preprocess'
OUTDIR="/home/tfuser/data/serialized_examples/"

#for COLL in top50 top51 top400 top401; do
#    for EXPRDIR in `ls -d $DATADIR/$COLL/* `; do
for COLL in top50; do
    for EXPRDIR in `ls -d $DATADIR/$COLL/hidden* `; do
        KEY=`echo $EXPRDIR | sed "s#$DATADIR/$COLL/##g"`
        echo $KEY
        SEROUTDIR="$OUTDIR/$COLL/$KEY"
        mkdir -p $SEROUTDIR
        LOGFILE="$SEROUTDIR/$KEY.log"
        CMD="python3 $CODEDIR/serialize_features.py $EXPRDIR $SEROUTDIR \
            --label_col 1957 >> $LOGFILE"
        echo $CMD
        echo $CMD > $LOGFILE
        eval $CMD
    done;
done
```

## Serialize for mcf10a_progression and tcga_paired_znorm
```
DATADIR='/mnt/knowdnn_hdd/tfuser/data/expression_examples/'
CODEDIR='/home/tfuser/code/preprocess'
OUTDIR="/home/tfuser/data/serialized_examples/"

for COLL in tcga_paired_znorm; do
    echo $COLL
    EXPRDIR="$DATADIR/$COLL/"
    SEROUTDIR="$OUTDIR/$COLL/"
    mkdir -p $SEROUTDIR
    LOGFILE="$SEROUTDIR/$COLL.log"
    CMD="python3 $CODEDIR/serialize_features.py $EXPRDIR $SEROUTDIR \
        --label_col 1957 >> $LOGFILE"
    echo $CMD
    echo $CMD > $LOGFILE
    eval $CMD
done
```
