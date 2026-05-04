function lazer_ses_veri_fuzyonu_veriseti()
% Tek dosyada: WAV (8x2048) + CSV (lazer) veri eşleştirme, öznitelik çıkarımı,
% erken/geç/hibrit füzyon ve temel sınıflandırma (NaN/Inf güvenli, 'omitnan'sız).

%% ======================== AYARLAR ========================
pairingMode = "samefolder"; % "samefolder" / "separate"

% SAMEFOLDER modu için:
sameFolderDir = "C:\Users\dogan\OneDrive\Desktop\lazer_ve_ses_sinyalleri\Data\lazer_ses_birlikte";

% SEPARATE modu için (gerekirse doldur):
audioRoot = "";
laserRoot = "";

laserWindowLen = 36; % lazer penceresi

%% ======================== HAZIRLIK ========================
switch pairingMode
    case "samefolder"
        D = dir(fullfile(sameFolderDir, "*.csv"));
        if isempty(D), error("CSV bulunamadı: %s", sameFolderDir); end
    case "separate"
        D = dir(fullfile(laserRoot, "*.csv"));
        if isempty(D), error("CSV bulunamadı: %s", laserRoot); end
    otherwise
        error("pairingMode hatalı.");
end

allX = [];
allY = [];
featNamesLaser = [];
featNamesAudio = [];

%% ======================== DÖNGÜ ==========================
for k = 1:numel(D)
    csvPath = fullfile(D(k).folder, D(k).name);
    [~, baseName] = fileparts(csvPath);

    switch pairingMode
        case "samefolder"
            wavPath = fullfile(D(k).folder, baseName + ".wav");
        case "separate"
            wavPath = fullfile(audioRoot, baseName + ".wav");
    end
    if ~isfile(wavPath)
        warning("Eş WAV yok, atlanıyor: %s", wavPath);
        continue
    end

    % ---- CSV (lazer) ----
    laser = readLaserAfterSemicolon(csvPath); % sütun vektörü
    if size(laser,2) > 1
        laser = mean_valid_cols(laser); % çok sütun ise satır ortalaması
    end
    laser = laser(:);
    nWin = floor(numel(laser)/laserWindowLen);
    if nWin == 0
        warning("Kısa lazer verisi: %s", csvPath); continue
    end
    laser = laser(1:nWin*laserWindowLen);
    laserWins = reshape(laser, laserWindowLen, nWin).'; % nWin x 36

    % Lazer öznitelikleri
    L_feat = zeros(nWin, numLaserFeatures_noomit());
    for i = 1:nWin
        [lvec, lnames] = extract_features_vector_noomit(laserWins(i,:));
        L_feat(i,:) = lvec;
    end
    if isempty(featNamesLaser), featNamesLaser = "L_" + lnames; end

    % ---- WAV (ses) ----
    [y, ~] = audioread(wavPath);
    % 2048x8 bekleniyor; 8x2048 gelirse döndür
    if size(y,1)==8 && size(y,2)==2048, y = y.'; end
    if size(y,2)~=8
        warning("8 kanal değil (%d): %s", size(y,2), wavPath);
    end
    if size(y,1)~=2048
        warning("2048 örnek değil (%d): %s", size(y,1), wavPath);
    end
    % Satır bazlı (örnek bazlı) kanal ortalaması (NaN/Inf güvenli)
    yMean = mean_valid_rows(y);  % 2048x1

    % Ses öznitelikleri (tek vektör)
    [A_feat1, anames] = extract_features_vector_noomit(yMean);
    if isempty(featNamesAudio), featNamesAudio = "A_" + anames; end
    A_feat = repmat(A_feat1(:).', nWin, 1); % her pencereye aynı ses özniteliği

    % ---- Füzyon + Etiket ----
    XY = [L_feat, A_feat];
    label = map_last_char_to_label(baseName);
    if isnan(label)
        warning("Son harf B/F/L/R değil, atlandı: %s", baseName);
        continue
    end
    Y_local = repmat(label, nWin, 1);

    allX = [allX; XY]; %#ok<AGROW>
    allY = [allY; Y_local]; %#ok<AGROW>
end

%% ======================== ÇIKTILAR ========================
if isempty(allX)
    error("Hiç örnek oluşmadı. Yol/isimleri kontrol et.");
end
featNames = [featNamesLaser, featNamesAudio];
featNames = matlab.lang.makeUniqueStrings(string(featNames));

T = array2table(allX, "VariableNames", cellstr(featNames));
T.label = allY;

if pairingMode=="samefolder"
    outDir = char(sameFolderDir);   % yazma fonksiyonlarıyla uyum için char’a çeviriyorum
else
    outDir = pwd;                   % çalışma dizini
end
save(fullfile(outDir, "dataset_fused.mat"), "allX", "allY", "T", "-v7.3");
writetable(T, fullfile(outDir, "dataset_fused.csv"));
fprintf("Bitti. Örnek: %d, Özellik: %d\n", size(allX,1), size(allX,2));
end

%% ====================== YARDIMCI FONKSİYONLAR ======================

function data = readLaserAfterSemicolon(csvPath)
% ';' sonrası lazer sütun(ları)
opts = detectImportOptions(csvPath, "Delimiter",";", "NumHeaderLines",0);
try
    M = readmatrix(csvPath, opts);
    if size(M,2) >= 2
        data = M(:,2:end);
        data = data(:, any(isfinite(data),1));
        if isempty(data), data = M(:,end); end
        return
    end
catch
end
% Manuel okuma
lines = readlines(csvPath);
vals = [];
for i=1:numel(lines)
    L = strtrim(lines(i)); if strlength(L)==0, continue; end
    p = split(L, ";"); if numel(p)<2, continue; end
    r = strtrim(p(2));
    if contains(r, ",")
        nums = str2double(split(r, ","));
    else
        nums = str2double(split(r));
    end
    nums = nums(~isnan(nums));
    if isempty(nums), continue; end
    vals = [vals; nums(:).']; %#ok<AGROW>
end
if isempty(vals), data = []; else, data = vals; end
end

function n = numLaserFeatures_noomit()
[vec, ~] = extract_features_vector_noomit(randn(1,36));
n = numel(vec);
end

function [vec, names] = extract_features_vector_noomit(x)
% NaN/Inf güvenli, 'omitnan' KULLANMAZ
x = x(:);
v = x(isfinite(x)); % geçerli değerler
if isempty(v)
    vec   = zeros(1,21);
    names = ["feat_mean","feat_std","feat_var","feat_median","feat_mad", ...
             "feat_min","feat_max","feat_ptp","feat_rms","feat_energy", ...
             "feat_slope","feat_skew","feat_kurt","feat_ac1","feat_ac2", ...
             "feat_spec_centroid","feat_spec_spread","feat_spec_entropy", ...
             "feat_dom_freq_bin","feat_dom_power","feat_rolloff85_bin"];
    return
end

% Temel istatistikler
mu   = mean(v);
sd   = std(v);
vr   = var(v);
med  = median(v);
madv = median(abs(v - med)); % MAD (b=1 ile)
mn   = min(v);
mx   = max(v);
ptp  = mx - mn;
rmsv = sqrt(mean(v.^2));
eng  = sum(v.^2);

% Doğrusal eğim
N = numel(v);
t = (0:N-1).';
if all(v==v(1)), slope = 0; else, p = polyfit(t, v, 1); slope = p(1); end

% Şekil ölçüleri (NaN yok; direkt)
sk = skewness(v);
ku = kurtosis(v);

% Otokorelasyon (lag1, lag2)
ac1 = safe_autocorr_noomit(v, 1);
ac2 = safe_autocorr_noomit(v, 2);

% Spektral öznitelikler
[sc, ss, se, db, dp, r85] = spectral_features_noomit(v);

vec = [mu, sd, vr, med, madv, mn, mx, ptp, rmsv, eng, ...
       slope, sk, ku, ac1, ac2, sc, ss, se, db, dp, r85];

names = ["feat_mean","feat_std","feat_var","feat_median","feat_mad", ...
         "feat_min","feat_max","feat_ptp","feat_rms","feat_energy", ...
         "feat_slope","feat_skew","feat_kurt","feat_ac1","feat_ac2", ...
         "feat_spec_centroid","feat_spec_spread","feat_spec_entropy", ...
         "feat_dom_freq_bin","feat_dom_power","feat_rolloff85_bin"];
end

function r = safe_autocorr_noomit(v, lag)
v = v(:);
N = numel(v);
if lag >= N, r = 0; return; end
v = v - mean(v);
den = sum(v.^2);
if den<=0 || ~isfinite(den), r = 0; return; end
num = sum( v(1:end-lag) .* v(1+lag:end) );
r = num / den;
end

function [cent, spr, ent, domBin, domPow, roll85Bin] = spectral_features_noomit(v)
v = v(:);
N = numel(v);
if N <= 1
    cent=0; spr=0; ent=0; domBin=1; domPow=0; roll85Bin=1; return
end
v = v - mean(v);

w = hamming(N);
xf = fft(v .* w);
P2 = (abs(xf).^2) / N;
P1 = P2(1:floor(N/2)+1);
if numel(P1) > 2, P1(2:end-1) = 2*P1(2:end-1); end

bins = (0:numel(P1)-1).';
P = P1(:);
Ps = sum(P);
if Ps <= 0 || ~isfinite(Ps)
    cent=0; spr=0; ent=0; domBin=1; domPow=0; roll85Bin=1; return
end

wmean = sum(bins .* P) / Ps;
cent  = wmean;
spr   = sqrt( sum(((bins - wmean).^2) .* P) / Ps );

p = P / Ps;
p(p<=0) = eps;
ent = -sum(p .* log(p));

[domPow, domBin] = max(P);

target = 0.85 * Ps;
cs = cumsum(P);
roll85Bin = find(cs >= target, 1, 'first');
if isempty(roll85Bin), roll85Bin = numel(P); end
end

function m = mean_valid_rows(Y)
% Her satırda finite değerlerin ortalaması (NaN/Inf güvenli)
Y = double(Y);
isv = isfinite(Y);
sumv = sum(Y .* isv, 2);
cnt  = sum(isv, 2);
cnt(cnt==0) = 1;
m = sumv ./ cnt;
end

function m = mean_valid_cols(Y)
% Her satır için çok sütun varsa sütun ortalaması (finite değerlerden)
isv = isfinite(Y);
sumv = sum(Y .* isv, 2);
cnt  = sum(isv, 2);
cnt(cnt==0) = 1;
m = sumv ./ cnt;
end

function y = map_last_char_to_label(baseName)
% B=1, F=2, L=3, R=4
ch = upper(extractAfter(baseName, strlength(baseName)-1));
if strlength(ch) ~= 1, ch = upper(baseName(end)); end
switch ch
    case 'B', y = 1;
    case 'F', y = 2;
    case 'L', y = 3;
    case 'R', y = 4;
    otherwise, y = NaN;
end
end
