%-----------------------------------------------------------------------
% Job saved on 16-Sep-2022 15:02:23 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
%%
matlabbatch{1}.spm.tools.cat.long.datalong.subjects = {
                                                       {
                                                       'sub-00001_ses-01_T1w.nii,1'
                                                       'sub-00001_ses-02_T1w.nii,1'
                                                       'sub-00001_ses-03_T1w.nii,1'
                                                       'sub-00001_ses-04_T1w.nii,1'
                                                        }
                                                       }';
%%
matlabbatch{1}.spm.tools.cat.long.longmodel = 1;
matlabbatch{1}.spm.tools.cat.long.enablepriors = 1;
matlabbatch{1}.spm.tools.cat.long.prepavg = 2;
matlabbatch{1}.spm.tools.cat.long.bstr = 0;
matlabbatch{1}.spm.tools.cat.long.avgLASWMHC = 0;
matlabbatch{1}.spm.tools.cat.long.nproc = 6; % Number of parallell processes. Modify according to you preferences/resources
matlabbatch{1}.spm.tools.cat.long.opts.tpm = {'/home/common/neuroimage/software/spm12/tpm/TPM.nii'};
matlabbatch{1}.spm.tools.cat.long.opts.affreg = 'mni';
matlabbatch{1}.spm.tools.cat.long.opts.biasacc = 0.5;
matlabbatch{1}.spm.tools.cat.long.extopts.restypes.optimal = [1 0.3];
matlabbatch{1}.spm.tools.cat.long.extopts.setCOM = 1;
matlabbatch{1}.spm.tools.cat.long.extopts.APP = 1070;
matlabbatch{1}.spm.tools.cat.long.extopts.affmod = 0;
matlabbatch{1}.spm.tools.cat.long.extopts.spm_kamap = 0;
matlabbatch{1}.spm.tools.cat.long.extopts.LASstr = 0.5;
matlabbatch{1}.spm.tools.cat.long.extopts.LASmyostr = 0;
matlabbatch{1}.spm.tools.cat.long.extopts.gcutstr = 2;
matlabbatch{1}.spm.tools.cat.long.extopts.WMHC = 2;
matlabbatch{1}.spm.tools.cat.long.extopts.registration.shooting.shootingtpm = {'/home/common/neuroimage/software/spm12/toolbox/cat12/templates_MNI152NLin2009cAsym/Template_0_GS.nii'};
matlabbatch{1}.spm.tools.cat.long.extopts.registration.shooting.regstr = 0.5;
matlabbatch{1}.spm.tools.cat.long.extopts.vox = 1.5;
matlabbatch{1}.spm.tools.cat.long.extopts.bb = 12;
matlabbatch{1}.spm.tools.cat.long.extopts.SRP = 22;
matlabbatch{1}.spm.tools.cat.long.extopts.ignoreErrors = 1;
matlabbatch{1}.spm.tools.cat.long.output.BIDS.BIDSno = 1;
matlabbatch{1}.spm.tools.cat.long.output.surface = 1;
matlabbatch{1}.spm.tools.cat.long.ROImenu.atlases.neuromorphometrics = 1;
matlabbatch{1}.spm.tools.cat.long.ROImenu.atlases.lpba40 = 0;
matlabbatch{1}.spm.tools.cat.long.ROImenu.atlases.cobra = 1;
matlabbatch{1}.spm.tools.cat.long.ROImenu.atlases.hammers = 0;
matlabbatch{1}.spm.tools.cat.long.ROImenu.atlases.thalamus = 1;
matlabbatch{1}.spm.tools.cat.long.ROImenu.atlases.suit = 1;
matlabbatch{1}.spm.tools.cat.long.ROImenu.atlases.ibsr = 0;
matlabbatch{1}.spm.tools.cat.long.ROImenu.atlases.ownatlas = {''};
matlabbatch{1}.spm.tools.cat.long.longTPM = 1;
matlabbatch{1}.spm.tools.cat.long.modulate = 1;
matlabbatch{1}.spm.tools.cat.long.dartel = 0;
matlabbatch{1}.spm.tools.cat.long.delete_temp = 1;
