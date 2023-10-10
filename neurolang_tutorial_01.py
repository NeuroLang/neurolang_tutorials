# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: neurolang_tutorial_310
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline

# %%
# %pip install -q git+https://github.com/wannesm/PySDD.git#egg=PySDD
# %pip install -q neurolang

# %%
from IPython.display import display
from neurolang.frontend import NeurolangPDL
from neurolang.utils.server.engines import NeurosynthEngineConf
import os
from pathlib import Path
import numpy as np
from typing import Callable
from nilearn import plotting, datasets
from matplotlib import pyplot as plt
import nibabel as nib

# %% [markdown]
# # NeuroLang is a logic language enabling rich meta-analyses.
#
# For instance in the next image, a meta-analysis of the Neurosynth database was able to pinpoint the
# most likely cognitive terms associated with different functional networks ([Abdallah et al [Sci Rep 2022](https://www.nature.com/articles/s41598-022-21801-4), [eLife 2022](https://elifesciences.org/articles/76926)).
#
#
# <img src="abdallah_et_al_fig2.png" width="750"/>
#

# %% [markdown]
# # Let's learn neurolang by extracting information from NeuroSynth!
#
#
# Let's start by creating NeuroLang instance. Specifically one with the Neurosynth database loaded. This database has three tables:
#
# * `TermInStudyTFIDF(term, tfidf, study)` which contains all terms mentioned in a study with their tfidf
# * `PeaksReported(x, y, z, study)` which contains the peaks, in MNI 152 space, reported in a study.
# * `Voxels(x, y, z)` which contains all voxels of the MNI 152 space, at 1mm resolution
# * `Study(s)` the set of all studies
# * `SelectedStudy(s)` which represents that every study in the database is an equiprobable token of neuroscientific knowledge. Specifically, the table represents the probabilistic events "one study `s` was selected" where each event has the probability `1 / <total number of Studies>` and these events are exclusive.
#
# The set of tables in a NeuroLang instance can be called the _extensional database_.

# %%
neurolang_engine_factory = NeurosynthEngineConf(Path(os.getcwd()))
nl = neurolang_engine_factory.create()

# %% [markdown]
# Let's make a first query obtaining all terms mentioned in all studies. Specifically we are going to encode the rule:
#
# Every `term` in the set `Term` is the first element of the tuples in the set `TermInStudyTFIDF`
#
#
# $\forall term: Term(term) \leftarrow TermInStudyTFIDF(term, tfidf, study)$
#
# and then we will add a line expressing that the expected answer is all the terms in the set `Term`
#

# %%
with nl.scope as e:
    res = nl.execute_datalog_program(
    """
        Term(term) :- TermInStudyTFIDF(term, tfidf, study)
        ans(term) :- Term(term)
    """
    )
    display(res)

# %% [markdown]
# Next, we will extract the top 99.999% most important terms according to their TFIDF. For this we will add a `numpy` function into our engine.
# Then, we will define the TFIDF threshold, and use it to select the terms with two rules representing
#
# "Define all Important Terms as those whose TFIDF is in the top 99.999%":
#
# * The TFIDFThreshold is the TFIDF valuee at the 99.999 percentile
# * The ImportanTerms are those such that there is at least one article where the TFIDF is larger than the threshold 
#   * $\forall term, tfidf: ImportantTerm(term, tfidf)\leftarrow \exists study: TermInStudy(term, tfidf, study) \wedge tfidf > thr \wedge TFIDFThreshold(thr)$
#

# %%
nl.add_symbol(
    np.percentile, name="percentile", type_=Callable[[float, float], float]
)
with nl.scope as e:
    res = nl.execute_datalog_program(
    """
        TFIDFThreshold(percentile(tfidf, 99.999)) :- TermInStudyTFIDF(t, tfidf, study)
        ImportantTerm(term, tfidf) :- TermInStudyTFIDF(term, tfidf, study) & (tfidf > thr) & TFIDFThreshold(thr)
        ans(term, tfidf) :- ImportantTerm(term, tfidf)
    """
    )
    display(res.as_pandas_dataframe().sort_values(by='tfidf'))


# %% [markdown]
# We can do the same thing but using more convenient expressions in the query

# %%
with nl.scope as e:
    res = nl.execute_datalog_program(
    """
        Threshold_tfidf(percentile(tfidf, (100 - .001))) :- TermInStudyTFIDF(t, tfidf, study)
        Term(t, tfidf) :- TermInStudyTFIDF(t, tfidf, study) & (tfidf > thr) & Threshold_tfidf(thr)
        ans(term, tfidf) :- Term(term, tfidf)
    """
    )
    display(res.as_pandas_dataframe().sort_values(by='tfidf'))


# %% [markdown]
# Next we are going to extract all MNI voxels that have been reported active in a study mentioning the term "pain".
#
# Specifically,
# 1. we will consider active those who are at most within 4mm of a peak reported in a study.
#      * "A Voxel is Reported in a study if there is a peak reported in such study, and the voxel is within 4mm of this peak
#
#
# 2. we will filter all the studies according to the importance of the word "pain" in the study
#      * "a term is in a pain study if the therm is mentioned in the study and it's TFIDF in the study is larger tham 0.1"
#
#
# 3. We will count how many times a voxel was reported in a pain study
#

# %%
with nl.scope as e:
    res = nl.execute_datalog_program(
    """
        VoxelReported (x, y, z, study) :- PeakReported(x2, y2, z2, study) & Voxel(x, y, z) & (d == EUCLIDEAN(x, y, z, x2, y2, z2)) & (d < 4)
        TermInEmotionStudy(term, study) :- TermInStudyTFIDF(term, tfidf, study) & (tfidf > 0.1)
        ReportedInEmotionStudy(x, y, z, count(study)) :- VoxelReported (x, y, z, study) & TermInEmotionStudy("pain", study)
        ans(x, y, z, s) :- ReportedInEmotionStudy(x, y, z, s)
    """
    )
    display(res)

# %% [markdown]
# # Creating overlays as results for the analysis
#
# The results as a voxel list are not very easy to understand. So we will agregate all voxel counts in an image

# %%
with nl.scope as e:
    res = nl.execute_datalog_program(
    """
        VoxelReported (x, y, z, study) :- PeakReported(x2, y2, z2, study) & Voxel(x, y, z) & (d == EUCLIDEAN(x, y, z, x2, y2, z2)) & (d < 4)
        TermInEmotionStudy(term, study) :- TermInStudyTFIDF(term, tfidf, study) & (tfidf > 0.1)
        ReportedInEmotionStudy(x, y, z, count(study)) :- VoxelReported (x, y, z, study) & TermInEmotionStudy("pain", study)
        EmotionCountImage(agg_create_region_overlay(x, y, z, s)) :- ReportedInEmotionStudy(x, y, z, s)
        ans(img) :- EmotionCountImage(img)
    """
    )
    display(res)
region = res.as_pandas_dataframe().iloc[0, 0]
thr = np.percentile(region.overlay, 95)
plotting.plot_img(
    region.spatial_image(), colorbar=True,
    threshold=thr, bg_img=datasets.load_mni152_template(),
    display_mode='y', cut_coords=np.linspace(-50, 50, 10)
)

# %% [markdown]
# ## Excercise: repeat the task but counting the a study reports a voxel when the study mentions "language" and "audition".

# %%

# %% [markdown]
# # Let's now talk about probabilities! What is the probability that a voxel is mentioned in a Study?
#
# Now let's formulate the same question but in a probabilistic manner. Specifically we will produce the probability that a voxel is mentioned in a study, given that the study mentions the word "emotion", in shorthand notation this is
# * $P(voxel \in Study | ``emotion" \in Study)$
# Which we can write in neurolang as:
# * `VoxelsGivenStudyMentionsEmotion(x, y, z, PROB) :- VoxelReported(x, y, z, study) // (TermInEmotionStudy("emotion", study) & SelectedStudy(study))`
#   In this case the `PROB` term in the left-hand side of the rule, or head, means that we will obtain the probability of the stochastic event on the right. The `//` operator means conditional probability, and `SelectedStudy` as we described at the beginning of the notebook, is a probabilistic event, denoting that the study `study` is mentioned with probability `1 / <total number of studies>.`

# %%
with nl.scope as e:
    res = nl.execute_datalog_program(
    """
        VoxelReported (x, y, z, study) :- PeakReported(x2, y2, z2, study) & Voxel(x, y, z) & (d == EUCLIDEAN(x, y, z, x2, y2, z2)) & (d < 4)
        TermInEmotionStudy(term, study) :- TermInStudyTFIDF(term, tfidf, study) & (tfidf > 0.1)
        VoxelsGivenStudyMentionsEmotion(x, y, z, PROB) :- VoxelReported(x, y, z, study) // (TermInEmotionStudy("emotion", study) & SelectedStudy(study))
        image(agg_create_region_overlay(x, y, z, p)) :- VoxelsGivenStudyMentionsEmotion(x, y, z, p)
        ans(img) :- image(img)
    """
    )
    display(res)
region = res.as_pandas_dataframe().iloc[0, 0]
thr = np.percentile(region.overlay, 95)
plotting.plot_stat_map(region.spatial_image(), threshold=thr, display_mode='y', cut_coords=np.linspace(-50, 50, 10))

# %% [markdown]
# Nonetheless, it's hard to know beforehand if the probabilities are meaninful. We need to compare them to a baseline.
#
# So let's compare the probability of a voxel being reported in a study given that it mentions the word "emotion", against the probability of being reported in a study that doesn't mention this work.

# %%
with nl.scope as e:
    res = nl.execute_datalog_program(
    """
        VoxelReported (x, y, z, study) :- PeakReported(x2, y2, z2, study) & Voxel(x, y, z) & (d == EUCLIDEAN(x, y, z, x2, y2, z2)) & (d < 2)

        TermInEmotionStudy(study) :- TermInStudyTFIDF("emotion", tfidf, study) & (tfidf > 0.01)
        TermInStudyNoEmotion(study) :-  Study(study) & ~TermInEmotionStudy(study)

        VoxelInEmotionStudy(x, y, z, PROB) :- VoxelReported (x, y, z, study) // (TermInEmotionStudy(study) & SelectedStudy(study))
        VoxelInNoEmotionStudy(x, y, z, PROB) :- VoxelReported (x, y, z, study) // (TermInStudyNoEmotion(study) & SelectedStudy(study))

        image(agg_create_region_overlay(x, y, z, p)) :- VoxelInEmotionStudy(x, y, z, p0) & VoxelInNoEmotionStudy(x, y, z, p1) & (p == log(p0 / p1))

        ans(img) :- image(img)
    """
    )
    display(res)
region = res.as_pandas_dataframe().iloc[0, 0]
plotting.plot_stat_map(region.spatial_image(), threshold=1, display_mode='y', cut_coords=np.linspace(-50, 50, 10))


# %% [markdown]
# ## Excercise:
#
# * Produce the image of the probability that a voxel will be mentioned "language" and "audition" being mentioned simultaneously on a study
# * Produce the image of the log likelihood ratios comparing the probability that a voxel will be mentioned in a study mentioning "language" and that it will be reported in a study mentioning "audition".

# %%

# %% [markdown]
# # Reverse Inference. Figuring out the cognitive terms associated to the Yeo 7 Network parcellation

# %%
# Make the atlas into a set of tuples (x, y, z, label)
yeo_atlas = datasets.fetch_atlas_yeo_2011()
yeo7 = nib.load(yeo_atlas['thin_7'])
plotting.plot_roi(yeo_atlas['thin_7'], colorbar=True)

yeo_data = yeo7.get_fdata()
yeo_ijk = np.nonzero(yeo_data)
yeo_xyz = nib.affines.apply_affine(yeo7.affine, np.transpose(yeo_ijk)[:, :3])
yeo_labels = yeo_data[yeo_ijk]
yeo7_melt = np.c_[yeo_xyz, yeo_labels]

# Add the Set to the neurolang engine as the "Yeo7" set

nl.add_tuple_set(yeo7_melt, name='Yeo7')

# %%
with nl.scope as e:
    res = nl.execute_datalog_program(
    """
        NetworkReported(network, study) :- Yeo7(x2, y2, z2, network) & PeakReported(x, y, z, study) & (d == EUCLIDEAN(x, y, z, x2, y2, z2)) & (d <= 2)
        Network(network) :- Yeo7(..., ..., ..., network) 
        TermInStudy(term, study) :- TermInStudyTFIDF(term, tfidf, study) & (tfidf > 0.1)
        NetworkTerm(term, network, PROB) :- TermInStudy(term, study) // (NetworkReported(network, study) & SelectedStudy(study))
        Term(term, PROB) :- TermInStudy(term, study) & SelectedStudy(study)
        NetworkLR(term, network, ll) :- NetworkTerm(term, network, p) & Term(trem, p1) & (ll == log(p / p1))
        ans(term, network, p) :- NetworkLR(term, network, p)
    """
    )
    display(res.as_pandas_dataframe().sort_values(by='p').groupby('network').tail(2).sort_values("network"))

# %%
with nl.scope as e:
    res = nl.execute_datalog_program(
    """
        NetworkReported(network, study) :- Yeo7(x2, y2, z2, network) & PeakReported(x, y, z, study) & (d == EUCLIDEAN(x, y, z, x2, y2, z2)) & (d <= 2)
        Network(network) :- Yeo7(..., ..., ..., network) 
        NetworkExclusiveReported(network, study) :- NetworkReported(network, study) & ~exists(n2; Network(n2) & (n2 != network) & NetworkReported(n2, study))
        TermInStudy(term, study) :- TermInStudyTFIDF(term, tfidf, study) & (tfidf > 0.1)
        NetworkTerm(term, network, PROB) :- TermInStudy(term, study) // (NetworkExclusiveReported(network, study) & SelectedStudy(study))
        Term(term, PROB) :- TermInStudy(term, study) & SelectedStudy(study)
        NetworkLR(term, network, ll) :- NetworkTerm(term, network, p) & Term(trem, p1) & (ll == log(p / p1))
        ans(term, network, p) :- NetworkLR(term, network, p)
    """
    )
    display(res.as_pandas_dataframe().sort_values(by='p').groupby('network').tail(2).sort_values("network"))

# %% [markdown]
# ## Excercise: Do the same but with the Yeo 17 network set

# %%
yeo_atlas = datasets.fetch_atlas_yeo_2011()
yeo17 = nib.load(yeo_atlas['thin_17'])
plotting.plot_roi(yeo_atlas['thin_17'], colorbar=True)

# %%

# %%
