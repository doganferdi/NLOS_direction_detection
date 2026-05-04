%% kodların doğru çalışması için lazer ve ses dosya adlarının aynı olması gerekiyor. sistemi buna göre kurmalısın



function fuse_audio_laser_train()
%% ================== KULLANICI AYARLARI ==================
audioRoot = "C:\Users\dogan\OneDrive\Desktop\lazer_ve_ses_sinyalleri\Data\Ses\Temiz";   % WAV kök klasörü
laserRoot = "C:\Users\dogan\OneDrive\Desktop\lazer_ve_ses_sinyalleri\Data\Lazer\Temiz"; % CSV kök klasörü
% Eşleştirme varsayımı: Aynı numaralandırma/kimlik şemasına sahip dosyalar birbirinin karşılığıdır.
% Eğer isimler birebir aynı değilse map eşlemesi yazın veya eşleştirme fonksiyonunu düzenleyin.

% Lazer penceresi ve öznitelikleri
laserWinLen = 36;       % 36 örnekli pencere
laserOverlap = 0;       % çakışmasız
rng(42);                % tekrarlanabilirlik

%% ================== 1) DOSYA LİSTESİ ====================
audioFiles = dir(fullfile(audioRoot, "**", "*.wav"));
if isempty(audioFiles)
    error("Ses dosyası bulunamadı: %s", audioRoot);
end

% İsteğe bağlı: CSV’ler için kontrol
laserFiles = dir(fullfile(laserRoot, "**", "*.csv"));
if isempty(laserFiles)
    error("Lazer dosyası bulunamadı: %s", laserRoot);
end

% Hızlı arama için harita
laserIndex = containers.Map('KeyType','char','ValueType','char');
for i=1:numel(laserFiles)
    [~,bn,~] = fileparts(laserFiles(i).name);
    laserIndex(lower(bn)) = fullfile(laserFiles(i).folder, laserFiles(i).name);
end

%% ================== 2) ÖZNİTELİK ÇIKARIMI ================
allAudioFeat = [];  % ses öznitelikleri
allLaserFeat = [];  % lazer öznitelikleri (dosya düzeyi özet)
allLabels    = [];  % yön etiketleri (categorical)
kept = 0; dropped = 0;

for k=1:numel(audioFiles)
    awav = fullfile(audioFiles(k).folder, audioFiles(k).name);
    [aPath,aBase,~] = fileparts(awav);
    label = parseDirectionFromName(aBase);
    if isnan(label)
        % Etiketi çözemiyorsak geç
        dropped = dropped + 1;
        continue;
    end

    % Karşılık gelen lazer csv’yi bulmaya çalış
    % Strateji: aynı temel ada yakın eşleşme; yoksa alt klasörde aynı numarayı arayın.
    candidateKeys = genCandidateKeys(aBase);
    csvPath = "";
    for c = 1:numel(candidateKeys)
        key = lower(candidateKeys{c});
        if isKey(laserIndex, key)
            csvPath = laserIndex(key);
            break;
        end
    end
    if csvPath == ""
        % birebir eşleşmedi; aynı klasörde benzer dosyayı arayalım (opsiyonel gevşek arama)
        % Yorum satırında bırakıldı: gerektiğinde özelleştirin.
        % fprintf("Uyarı: %s için lazer csv bulunamadı.\n", aBase);
        dropped = dropped + 1;
        continue;
    end

    % ---- SES ÖZNİTELİKLERİ ----
    try
        audioFeat = extractAudioFeatures(awav);
    catch ME
        warning("Ses öznitelik hatası (%s): %s", aBase, ME.message);
        dropped = dropped + 1;
        continue;
    end

    % ---- LAZER ÖZNİTELİKLERİ ----
    try
        laserFeat = extractLaserFeaturesFileLevel(csvPath, laserWinLen, laserOverlap);
    catch ME
        warning("Lazer öznitelik hatası (%s): %s", aBase, ME.message);
        dropped = dropped + 1;
        continue;
    end

    if any(isnan(audioFeat)) || any(isnan(laserFeat))
        dropped = dropped + 1;
        continue;
    end

    allAudioFeat(end+1, :) = audioFeat;    %#ok<AGROW>
    allLaserFeat(end+1, :) = laserFeat;    %#ok<AGROW>
    allLabels(end+1,1)     = label;        %#ok<AGROW>
    kept = kept + 1;
end

fprintf("Eşleşen örnek sayısı: %d | Atlanan: %d\n", kept, dropped);
if kept < 10
    error("Çok az eşleşen örnek var. Eşleştirme mantığını kontrol edin.");
end

%% ================== 3) FÜZYON VE MODELLEME ===============
% Erken füzyon (özellik birleştirme)
X_early = [allAudioFeat, allLaserFeat];
Y = categorical(allLabels);

% Standartlaştırma
[X_early, muE, sigmaE] = zscore(X_early);
[XA, muA, sigmaA] = zscore(allAudioFeat);
[XL, muL, sigmaL] = zscore(allLaserFeat);

% Eğitim/Test ayır
cv = cvpartition(Y, 'Holdout', 0.2);
trainIdx = training(cv); testIdx = test(cv);

% ---- (A) Erken füzyon: SVM (ECOC) ----
mdlEarly = fitcecoc(X_early(trainIdx,:), Y(trainIdx), ...
    'Learners', templateSVM('KernelFunction','rbf','KernelScale','auto'), ...
    'Coding','onevsall','ClassNames',categories(Y));

predEarly = predict(mdlEarly, X_early(testIdx,:));
accEarly = mean(predEarly == Y(testIdx));
fprintf("[Erken Füzyon][SVM] Doğruluk: %.2f%%\n", 100*accEarly);
figure('Name','ConfusionChart - Early Fusion');
confusionchart(Y(testIdx), predEarly, 'Title', sprintf('Erken Füzyon (SVM) - Acc=%.2f%%',100*accEarly));

% ---- (B) Geç füzyon: ayrı modeller + olasılık birleştirme ----
% Ses modeli
mdlA = fitcensemble(XA(trainIdx,:), Y(trainIdx), 'Method','Bag'); % Bagged trees güçlüdür
[~,scoreA] = predict(mdlA, XA(testIdx,:));

% Lazer modeli
mdlL = fitcensemble(XL(trainIdx,:), Y(trainIdx), 'Method','Bag');
[~,scoreL] = predict(mdlL, XL(testIdx,:));

% Yumuşak oylama (eşit ağırlık)
scoresLate = scoreA + scoreL;
[~,mxIdx] = max(scoresLate, [], 2);
classes = mdlA.ClassNames;              % aynı sınıf sırası
predLate = classes(mxIdx);
accLate = mean(predLate == Y(testIdx));
fprintf("[Geç Füzyon][Bag+Bag] Doğruluk: %.2f%%\n", 100*accLate);
figure('Name','ConfusionChart - Late Fusion');
confusionchart(Y(testIdx), predLate, 'Title', sprintf('Geç Füzyon (Bag+Bag) - Acc=%.2f%%',100*accLate));

% ---- (C) Hibrit: PCA ile boyut indir + erken füzyon ----
[coeffA, XA_pca, ~, ~, expA] = pca(XA(trainIdx,:));
[coeffL, XL_pca, ~, ~, expL] = pca(XL(trainIdx,:));
kA = find(cumsum(expA) >= 95, 1, 'first');  % %95 varyans
kL = find(cumsum(expL) >= 95, 1, 'first');
XA_train_p = (XA(trainIdx,:) - mean(XA(trainIdx,:)))./std(XA(trainIdx,:),0,1) * coeffA(:,1:kA);
XL_train_p = (XL(trainIdx,:) - mean(XL(trainIdx,:)))./std(XL(trainIdx,:),0,1) * coeffL(:,1:kL);
XH_train = [XA_train_p, XL_train_p];

% test tarafı
XA_test_p = (XA(testIdx,:) - mean(XA(trainIdx,:)))./std(XA(trainIdx,:),0,1) * coeffA(:,1:kA);
XL_test_p = (XL(testIdx,:) - mean(XL(trainIdx,:)))./std(XL(trainIdx,:),0,1) * coeffL(:,1:kL);
XH_test = [XA_test_p, XL_test_p];

mdlHybrid = fitcecoc(XH_train, Y(trainIdx), ...
    'Learners', templateSVM('KernelFunction','rbf','KernelScale','auto'), ...
    'Coding','onevsall','ClassNames',categories(Y));
predHybrid = predict(mdlHybrid, XH_test);
accHybrid = mean(predHybrid == Y(testIdx));
fprintf("[Hibrit PCA+Erken][SVM] Doğruluk: %.2f%% (kA=%d, kL=%d)\n", 100*accHybrid, kA, kL);
figure('Name','ConfusionChart - Hybrid Fusion');
confusionchart(Y(testIdx), predHybrid, ...
    'Title', sprintf('Hibrit (PCA+SVM) - Acc=%.2f%% | kA=%d, kL=%d',100*accHybrid,kA,kL));

%% ================== 4) MODEL KAYDETME ===================
save("fusion_models.mat", ...
     "mdlEarly","muE","sigmaE", ...
     "mdlA","muA","sigmaA", ...
     "mdlL","muL","sigmaL", ...
     "mdlHybrid","coeffA","coeffL","kA","kL");

disp("Bitti. Modeller fusion_models.mat dosyasına kaydedildi.");
end

%% ================== YARDIMCI FONKSİYONLAR ==================

function label = parseDirectionFromName(basename)
% Dosya adından yön etiketini (front/back/right/left) çözer.
% Destek: 'front','back','left','right','on','arka','sag','sol','F','B','R','L'
s = lower(basename);
label = NaN;
% açık kelimeler
if contains(s,"front") || contains(s,"ön") || contains(s,"on")
    label = categorical("front");
elseif contains(s,"back") || contains(s,"arka") || contains(s,"behind")
    label = categorical("back");
elseif contains(s,"right") || contains(s,"sag") || contains(s,"sağ")
    label = categorical("right");
elseif contains(s,"left") || contains(s,"sol")
    label = categorical("left");
else
    % tek harfli kodlar (alt çizgi/delim arası)
    tokens = regexp(s, '(^|[^a-z])([fbrl])([^a-z]|$)', 'tokens');
    if ~isempty(tokens)
        ch = tokens{1}{2};
        switch ch
            case 'f', label = categorical("front");
            case 'b', label = categorical("back");
            case 'r', label = categorical("right");
            case 'l', label = categorical("left");
        end
    end
end
end

function keys = genCandidateKeys(aBase)
% Aynı örnek için lazer CSV adları bazen son ekleri farklı olabilir.
% Burada birkaç varyasyon üretip eşleştirmeyi kolaylaştırıyoruz.
keys = unique(lower([ ...
    string(aBase), ...
    erase(string(aBase), ["_wav","-wav"," wav"]), ...
    regexprep(string(aBase), '\.ch\d+$','') ...
]));
keys = cellstr(keys);
end

function feat = extractAudioFeatures(wavPath)
% Çok kanallı (N×C) ses -> sabit uzunluklu öznitelik vektörü
[x, fs] = audioread(wavPath);     % x: samples × channels
if size(x,1) < 256
    % çok kısa ise pad
    x = padarray(x, [256-size(x,1), 0], 'post');
end
C = size(x,2);
% Çerçeve parametreleri (çok kısa sinyaller için kısa pencereler)
wlen = min(256, size(x,1));
hop  = max(1, floor(wlen/2));

% Kanal başına öznitelikler: MFCC(13) ort/ss, RMS, ZCR, spektral centroid/spread, rolloff(0.85), entropi
allCh = [];
for c=1:C
    xc = x(:,c);

    % MFCC
    try
        coeffs = mfcc(xc, fs, 'LogEnergy','Ignore', 'WindowLength',wlen, 'OverlapLength',wlen-hop, 'NumCoeffs',13);
    catch
        % Audio Toolbox yoksa basit bir MFCC alternatifi yapamayız; fallback: DCT-log mel yerine istatistikler
        coeffs = zeros(max(1,floor((size(xc,1)-wlen)/(hop)+1)), 13);
    end
    mfcc_mean = mean(coeffs,1,"omitnan");
    mfcc_std  = std(coeffs,0,1,"omitnan");

    % RMS & ZCR
    r = buffer(xc, wlen, wlen-hop, "nodelay");
    r_rms = sqrt(mean(r.^2,1));
    r_zcr = zeroCrossRate(xc, wlen, hop);

    % Spektral ölçüler
    win = hann(wlen,"periodic");
    [S,F,T] = stft(xc, fs, 'Window',win, 'OverlapLength',wlen-hop, 'FFTLength',max(512,2^nextpow2(wlen)));
    P = abs(S).^2 + eps;
    % normalize per frame
    Pn = P ./ sum(P,1);

    % centroid & spread (Hz)
    freq = F(:);
    sc = sum(freq.*Pn,1);
    ss = sqrt(sum(((freq - sc).^2).*Pn,1));
    % rolloff 0.85
    csum = cumsum(Pn,1);
    rollIdx = arrayfun(@(k) find(csum(:,k)>=0.85,1,'first'), 1:size(Pn,2));
    rollHz = F(rollIdx);
    % entropy
    sent = -sum(Pn.*log2(Pn),1);

    feats = [ ...
        mfcc_mean, mfcc_std, ...
        mean(r_rms,"omitnan"), std(r_rms,0,"omitnan"), ...
        mean(r_zcr,"omitnan"), std(r_zcr,0,"omitnan"), ...
        mean(sc,"omitnan"),    std(sc,0,"omitnan"), ...
        mean(ss,"omitnan"),    std(ss,0,"omitnan"), ...
        mean(rollHz,"omitnan"),std(rollHz,0,"omitnan"), ...
        mean(sent,"omitnan"),  std(sent,0,"omitnan") ...
    ];
    allCh = [allCh, feats]; %#ok<AGROW>
end

% Kanal bazında birleştirme stratejisi: (a) kanalları ardışık ekle (daha zengin) veya
% (b) kanallar üzerinde ort/ss al (daha kompakt). Şu an (a) uygulanıyor -> allCh.
feat = allCh;
end

function zcr = zeroCrossRate(x, wlen, hop)
% Basit ZCR hesaplayıcı
N = length(x);
idx = 1:hop:(N-wlen+1);
zcr = zeros(1, numel(idx));
for i=1:numel(idx)
    seg = x(idx(i):idx(i)+wlen-1);
    zcr(i) = sum(abs(diff(sign(seg))))/(2*numel(seg));
end
end

function featFile = extractLaserFeaturesFileLevel(csvPath, winLen, overlap)
% Lazer CSV’yi oku, 36’lık pencerelerden öznitelik çıkar, dosya düzeyinde özetle
% CSV biçimi: ilk sütun zaman/stamp olabilir; sinyal sütununu seçiyoruz.
raw = readmatrix(csvPath, 'OutputType','double');
if isempty(raw)
    error("Boş csv: %s", csvPath);
end

% Sinyal sütun seçimi: noktalı-virgüllü yapıdaysa readmatrix çözer; genelde 2. sütun sinyal
if size(raw,2) == 1
    sig = raw(:,1);
else
    % çoğu önceki konuşmada ';' sonrası sinyal sütunu: 2. sütun varsayıyoruz
    sig = raw(:, end);  % gerektiğinde '2' yapın
end

sig = sig(:);
% uçtaki NaN/0-only pencereleri ayıkla (isteğe bağlı)
sig = sig(~isnan(sig));

% Pencereleme
step = winLen - overlap;
nFrames = floor((numel(sig)-winLen)/step) + 1;
if nFrames < 3
    % sinyal çok kısa ise gerekirse pad
    sig = [sig; zeros(winLen*3,1)];
    nFrames = floor((numel(sig)-winLen)/step) + 1;
end

featList = [];
for i=1:nFrames
    s = sig((1:winLen)+(i-1)*step);

    % Basit ama etkili pencere öznitelikleri
    f_mean   = mean(s);
    f_std    = std(s);
    f_rms    = rms(s);
    f_ptp    = peak2peak(s);
    f_skew   = skewness(s);
    f_kurt   = kurtosis(s);
    % Enerji & eğim
    f_energy = sum(s.^2);
    t = (1:winLen).';
    p = polyfit(t, s, 1);
    f_slope  = p(1);

    % AC(1) (otkorelasyon ilk gecikme)
    s0 = s - mean(s);
    ac1 = sum(s0(1:end-1).*s0(2:end)) / (sum(s0.^2)+eps);

    featList(i,:) = [f_mean, f_std, f_rms, f_ptp, f_skew, f_kurt, f_energy, f_slope, ac1]; %#ok<AGROW>
end

% Dosya düzeyi özet: (mean, std, p10, p90) -> sabit uzunluk
featFile = [ ...
    mean(featList,1,"omitnan"), ...
    std(featList,0,1,"omitnan"), ...
    prctile(featList,10,1), ...
    prctile(featList,90,1) ...
];
end
