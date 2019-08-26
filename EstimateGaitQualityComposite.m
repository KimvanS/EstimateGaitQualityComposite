function [GaitQualityComposite, GaitGualityMeas] = EstimateGaitQualityComposite(AccLoco, FS)
% GETGAITQUALITYCOMPOSITE - function to estimate gait quality characteristics
% Calculates stride regularity, RMS, index of harmonicity and power at 
% stride frequency based on trunk accelerometry using methods developed 
% between September 2010 and July 2015 by Kim van Schooten and Sietse 
% Rispens in the FARAO project (VU University Amsterdam)
%
% Syntax: [GaitQualityComposite GaitGualityMeas] = EstimateGaitQualityComposite(AccLoco, FS)
%
% Inputs:
%   AccLoco:                 Trunk accelerations during locomotion episode in VT, ML, AP directions in m/s^2
%   FS:                      Sample frequency of the AccLoco in samples/s
%
% Outputs:
%   GaitQualityComposite:    Gait quality composite score based on autocorrelation at dominant frequency in VT, standard deviation of the signal in ML, index of harmoncity in ML and power at dominant frequency in AP
%   GaitGualityMeas:         Structure containing all characteristics calculated here as fields
%
% Subfunctions: ParabolaVertex
%
% Key references: 
%   Rispens, van Schooten, Pijnappels, Daffertshofer, Beek, & van Dieen (2015). Identification of fall risk predictors in daily life measurements: gait characteristics’ reliability and association with self-reported fall history. NNR, 29(1), 54-61.
%   van Schooten, Pijnappels, Rispens, Elders, Lips, & van Dieën (2015). Ambulatory fall-risk assessment: amount and quality of daily-life gait predict falls in older adults. JGMS, 70(5), 608-615.
%   Van Schooten, Pijnappels, Rispens, Elders, Lips, Daffertshofer, Beek, & Van Dieen (2016). Daily-life gait quality as predictor of falls in older people: a 1-year prospective cohort study. PLoS one, 11(7), e0158623.
%
% Author: Kim van Schooten
% Neuroscience Research Australia, 139 Barker Street, Randwick 2031 NSW, Australia 
% email: kim.vanschooten@gmail.com
% Website: http://sites.google.com/site/kimvanschooten/
% July 2015; Last revision: 26-Aug-2019

% Copyright (C) 2019  KS van Schooten
% This program is free software: you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation, either version 3 of the License, or (at your
% option) any later version.
% 
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
% Public License for more details.
% 
% You should have received a copy of the GNU General Public License along
% with this program.  If not, see <https://www.gnu.org/licenses/>.

%% Set some parameters
WindowLen = round(FS*10);    % Minimum length for measures estimation and window length for PSD
StrideTimeRange = [0.4 4.0]; % Range to search for stride time (seconds)
N_Harm = 20;                 % number of harmonics used for harmonic ratio and index of harmonicity

%% Init output variable
GaitGualityMeas = struct();
GaitQualityComposite = nan;

%% Only do further processing if time series is long enough
if size(AccLoco,1) < WindowLen
    return
end

%% Calculate stride regularity from autocorrelations inspired by Moe-Nilssen & Helbostad (2005). Interstride trunk acceleration variability but not step width variability can differentiate between fit and frail older adults. Gait Posture. 21(2), 164-170.
RangeStart = round(FS*StrideTimeRange(1));
RangeEnd = round(FS*StrideTimeRange(2));
[Autocorr3x3,Lags] = xcov(AccLoco, RangeEnd, 'unbiased');
Autocorr = Autocorr3x3(:,[1 5 9]);
AutocorrSum = sum(Autocorr,2); % This sum is independent of sensor re-orientation, as long as axes are kept orthogonal

% check that autocorrelations are positive for any direction,
% i.e. the 3x3 matrix is positive-definite in the extended sense for
% non-symmetric matrices, meaning that M+M' is positive-definite,
% which is true if the determinants of all square upper left corner
% submatrices of M+M' are positive (Sylvester's criterion)
IXRange = (numel(Lags)-(RangeEnd-RangeStart)):numel(Lags);
AutocorrPlusTrans = Autocorr3x3+Autocorr3x3(:,[1 4 7 2 5 8 3 6 9]);
IXRangeNew = IXRange( ...
    AutocorrPlusTrans(IXRange,1) > 0 ...
    & prod(AutocorrPlusTrans(IXRange,[1 5]),2) > prod(AutocorrPlusTrans(IXRange,[2 4]),2) ...
    & prod(AutocorrPlusTrans(IXRange,[1 5 9]),2) + prod(AutocorrPlusTrans(IXRange,[2 6 7]),2) + prod(AutocorrPlusTrans(IXRange,[3 4 8]),2) ...
    > prod(AutocorrPlusTrans(IXRange,[1 6 8]),2) + prod(AutocorrPlusTrans(IXRange,[2 4 9]),2) + prod(AutocorrPlusTrans(IXRange,[3 5 7]),2) ...
    );
if isempty(IXRangeNew)
    GaitGualityMeas.StrideRegularity = nan(1,3);
else
    StrideTimeSamples = Lags(IXRangeNew(AutocorrSum(IXRangeNew)==max(AutocorrSum(IXRangeNew))));
    GaitGualityMeas.StrideRegularity = Autocorr(Lags==StrideTimeSamples,:)./Autocorr(Lags==0,:);
end

%% Calculate RMS
GaitGualityMeas.StandardDeviation = std(AccLoco,0,1);

%% Calculate index of harmonicity inspired by Lamoth, Beek & Meijer (2002). Pelvis–thorax coordination in the transverse plane during gait. Gait Posture, 16(2), 101-114.
% Get power spectra of detrended accelerations
AccLocDetrend = detrend(AccLoco);
P=zeros(0,size(AccLocDetrend,2));
for i=1:size(AccLocDetrend,2)
    [P1,~] = pwelch(AccLocDetrend(:,i),hamming(WindowLen),[],WindowLen,FS);
    [P2,F] = pwelch(AccLocDetrend(end:-1:1,i),hamming(WindowLen),[],WindowLen,FS);
    P(1:numel(P1),i) = (P1+P2)/2;
end
dF = F(2)-F(1);

% Obtain stride frequency using a different method (PSD vs autocorr) than in stride regularity above
% Set parameters
HarmNr = [2 1 2];
CommonRange = [0.6 1.2];
% Get modal frequencies and the 'mean freq. of the peak'
for i=1:3
    MF1I = find([zeros(5,1);P(6:end,i)]==max([zeros(5,1);P(6:end,i)]),1);
    MF1 = F(MF1I,1);
    IndAround = F>=MF1*0.5 & F<=MF1*1.5;
    MeanAround = mean(P(IndAround,i));
    PeakBeginI = find(IndAround & F<MF1 & P(:,i) < mean([MeanAround,P(MF1I,i)]),1,'last');
    PeakEndI = find(IndAround & F>MF1 & P(:,i) < mean([MeanAround,P(MF1I,i)]),1,'first');
    if isempty(PeakBeginI), PeakBeginI = find(IndAround,1,'first'); end
    if isempty(PeakEndI), PeakEndI = find(IndAround,1,'last'); end
    ModalF(i) = sum(F(PeakBeginI:PeakEndI,1).*P(PeakBeginI:PeakEndI,i))/sum(P(PeakBeginI:PeakEndI,i));
end
% Get stride frequency from modal frequencies after doing some checks
StrFreqFirstGuesses = ModalF./HarmNr;
StdOverMean = std(StrFreqFirstGuesses)/mean(StrFreqFirstGuesses);
StrideFrequency1 = median(StrFreqFirstGuesses);
if StrideFrequency1 > CommonRange(2) && min(StrFreqFirstGuesses) < CommonRange(2) && min(StrFreqFirstGuesses) > CommonRange(1)
    StrideFrequency1 = min(StrFreqFirstGuesses);
end
if StrideFrequency1 < CommonRange(1) && max(StrFreqFirstGuesses) > CommonRange(1) && max(StrFreqFirstGuesses) < CommonRange(2)
    StrideFrequency1 = min(StrFreqFirstGuesses);
end
HarmGuess = ModalF/StrideFrequency1;
StdHarmGuessRoundErr = std(HarmGuess - round(HarmGuess));
if StdOverMean < 0.1
    StrideFrequency = mean(StrFreqFirstGuesses);
else
    if StdHarmGuessRoundErr < 0.1 && all(round(HarmGuess) >= 1)
        StrideFrequency = mean(ModalF./round(HarmGuess));
    else
        StrideFrequency = StrideFrequency1;
    end
end

% Calculate the measure per separate dimension
for i=1:size(P,2)
    % Relative cumulative power and frequencies that correspond to these cumulative powers
    PCumRel = cumsum(P(:,i))/sum(P(:,i));
    FCumRel = F+0.5*dF;
    
    % Calculate relative power of first twenty harmonics, taking the power
    % of each harmonic with a band of + and - 10% of the first
    % harmonic around it
    PHarm = zeros(N_Harm,1);
    for Harm = 1:N_Harm
        FHarmRange = (Harm+[-0.1 0.1])*StrideFrequency;
        PHarm(Harm) = diff(interp1(FCumRel,PCumRel,FHarmRange));
    end
    
    % Derive index of harmonicity
    if i == 2 % for ML we expect odd instead of even harmonics
        GaitGualityMeas.IndexHarmonicity(i) = PHarm(1)/sum(PHarm(1:2:12));
    else
        GaitGualityMeas.IndexHarmonicity(i) = PHarm(2)/sum(PHarm(2:2:12));
    end
end

%% Get power at dominant frequency
N_Windows_Weiss = floor(size(AccLoco,1)/WindowLen);
N_SkipBegin_Weiss = ceil((size(AccLoco,1)-N_Windows_Weiss*WindowLen)/2);
PWwin = 200;
Nfft = 2^(ceil(log(WindowLen)/log(2)+1));
for WinNr = 1:N_Windows_Weiss
    AccWin = AccLoco(N_SkipBegin_Weiss+(WinNr-1)*WindowLen+(1:WindowLen),:);
    for i=1:3
        clear P;
        AccWin_i = (AccWin(:,i)-mean(AccWin(:,i)))/std(AccWin(:,i)); % normalize window
        [P,F]=pwelch(AccWin_i,PWwin,[],Nfft,FS);
        IXFRange = find(F>=0.5 & F<= 3);
        FDindClosest = IXFRange(find(P(IXFRange)==max(P(IXFRange)),1,'first'));
        FDAmp = P(FDindClosest);
        if FDindClosest ~= min(IXFRange) && FDindClosest ~= max(IXFRange)
            VertexIX = [-1 0 1] + FDindClosest;
            [FD,FDAmp] = ParabolaVertex(F(VertexIX),P(VertexIX));
        end
        ArrayFDAmp(WinNr,i) = FDAmp;
    end
end
GaitGualityMeas.WeissAmplitude = nanmedian(ArrayFDAmp,1);

%% calculate GaitQualityComposite as reported in van Schooten, Pijnappels, Rispens, Elders, Lips, Daffertshofer, Beek, & Van Dieen (2016). Daily-life gait quality as predictor of falls in older people: a 1-year prospective cohort study. PLoS one, 11(7), e0158623. with one minor correction (changed sign to facilitate interpretation)
% positive values indicate better quality & lower risk of falls, in original dataset the range of this compositescore is [-2.5 to 2.5], the mean is 0 and standard deviation is 1
GaitQualityforComposite = [GaitGualityMeas.StrideRegularity(1) GaitGualityMeas.StandardDeviation(2) GaitGualityMeas.IndexHarmonicity(2) GaitGualityMeas.WeissAmplitude(3)]; % select autocorrelation at dominant frequency in VT, standard deviation of the signal in ML, index of harmoncity in ML and power at dominant frequency in AP
GaitQuality = (GaitQualityforComposite-repmat([0.4537 1.2124 0.4838 0.5176], size(GaitQualityforComposite,1),1)) ./ repmat([0.1591 0.2708 0.2228 0.1231],size(GaitQualityforComposite,1),1); % Rescale gait quality characteristics to z-scores using mean and std from van Schooten et al. 2016
GaitQualityComposite = -1 * sum(GaitQuality.*[-0.718286325476242 0.175229558047312 0.268658378355463 -0.200339103183014],2); % Regression coefficients determined in van Schooten et al. 2016


function [xvert,yvert] = ParabolaVertex(x,y)
% find vertex (maximum or mnimum) of parabola
if numel(x)~=3 || numel(y)~=3 || numel(unique(x))~=3
    error ('x and y must be 3-element vectors, and x must contain 3 unique elements');
end

abc = [x(:).^2 x(:) ones(3,1)]\y(:);

xvert = -abc(2)/abc(1)/2;
if nargout>1
    yvert = [xvert.^2 xvert 1]*abc;
end
