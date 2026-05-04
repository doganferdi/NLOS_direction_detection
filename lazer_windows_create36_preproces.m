%% Lazer CSV -> 48'lik örnekler + sayısal etiket (0/1/2) tablo üretimi
% Ayarlar
anaklasor     = "C:\Users\dogan\Desktop\lazer_ve_ses_sinyalleri\veriseti\Data\Lazer\Temiz\Duvar\deneme";
penceere  = 36;                % 48'lik dilimler
veriboyutu_ = 16220;             % beklenen satır sayısı (bilgi amaçlı)
kayit_36_pencere_csv = fullfile(anaklasor, "dataset_lazer_36seg_labels012.csv");
kayit_36_pencere_mat_dosyasi = fullfile(anaklasor, "dataset_lazer_36seg_labels012.mat");

% Tüm csv'leri (alt klasörler dahil) topla
dosyalar_ = dir(fullfile(anaklasor, '**', '*.csv'));
if isempty(dosyalar_)
    error('Alt klasörlerde CSV bulunamadı: %s', anaklasor);
end

% Özellik adları
featNames = "f" + string(1:penceere);

% Sonuç biriktiriciler
allChunks = {};   % [Ni x 48] numeric
allLabels = {};   % [Ni x 1]  double/int

fprintf('Toplam %d dosya bulundu. İşleniyor...\n', numel(dosyalar_));

for k = 1:numel(dosyalar_)
    fpath = fullfile(dosyalar_(k).folder, dosyalar_(k).name);
    [~, baseName, ~] = fileparts(fpath);

    % ---- Veri okuma: ';' sonrası numerik kolonu al ----
    % Çoğu dosya için readmatrix yeterli olur
    M = readmatrix(fpath, 'Delimiter',';', 'OutputType','double');
    if size(M,2) >= 2
        vals = M(:,2);
    else
        % Alternatif okuyucu: readtable
        T = readtable(fpath, 'Delimiter',';', 'ReadVariableNames', false);
        if width(T) < 2
            warning('Beklenen formatta değil, atlanıyor: %s', fpath);
            continue;
        end
        vals = T{:,2};
        if ~isnumeric(vals), vals = str2double(string(vals)); end
    end

    % Uzunluk bilgisi (sadece uyarı)
    if numel(vals) ~= veriboyutu_
        fprintf(2, 'Uyarı: %s beklenen %d yerine %d satır.\n', baseName, veriboyutu_, numel(vals));
    end

    % Varsa NaN'leri doldur
    vals = vals(:);
    if any(isnan(vals))
        vals = fillmissing(vals,'linear','EndValues','nearest');
    end

    % 48'lik dilimlere böl, artanı at
    nSeg = floor(numel(vals) / penceere);
    if nSeg == 0
        warning('48''lik segment oluşturulamadı, atlanıyor: %s', fpath);
        continue;
    end
    vals = vals(1:nSeg*penceere);
    X    = reshape(vals, penceere, nSeg).';   % [nSeg x 48]

    % Sayısal etiket (a->0, o->1, c->2)
    y = getLabel012FromName(baseName);
    if isnan(y)
        fprintf(2, 'Uyarı: Sınıf etiketi (a/o/c) bulunamadı, atlanıyor: %s\n', baseName);
        continue;
    end
    labels = repmat(y, nSeg, 1);

    % Biriktir
    allChunks{end+1} = X;      %#ok<SAGROW>
    allLabels{end+1} = labels; %#ok<SAGROW>
end

% Birleştir ve tabloya dök
if isempty(allChunks)
    error('Hiç örnek oluşturulamadı. Dosya adlarını ve klasör yapısını kontrol edin.');
end

Xall = vertcat(allChunks{:});          % [N_total x 48]
Yall = vertcat(allLabels{:});          % [N_total x 1]
Tab  = array2table(Xall, 'VariableNames', cellstr(featNames));
Tab.Label = int32(Yall);

% Özet
fprintf('Toplam örnek sayısı: %d (her biri 48 özellik).\n', height(Tab));
fprintf('Sınıf dağılımı (Label=0/1/2):\n');
disp(groupsummary(Tab,"Label"))

% Kaydet
writetable(Tab, kayit_36_pencere_csv);
save(kayit_36_pencere_mat_dosyasi, 'Tab');

fprintf('Bitti.\nCSV: %s\nMAT: %s\n', kayit_36_pencere_csv, kayit_36_pencere_mat_dosyasi);

%% ----------------- Yardımcı fonksiyon -----------------
function y = getLabel012FromName(fname)
% Dosya adındaki 'a','o','c' harflerine göre sayısal etiket döndürür.
% a->0 (ayakta), o->1 (oturuyor), c->2 (çömelmiş)
nameLower = lower(fname);

% Önce tek harfli tokenlara bak (_a_, -o-, vb.)
tokens = regexp(nameLower, '[a-z]+', 'match');
y = NaN;

if any(strcmp(tokens, 'a')), y = 0; return; end
if any(strcmp(tokens, 'o')), y = 1; return; end
if any(strcmp(tokens, 'c')), y = 2; return; end

% Daha gevşek: harf sınırlarıyla arama
if ~isempty(regexp(nameLower, '(^|[^a-z])a([^a-z]|$)', 'once')), y = 0; return; end
if ~isempty(regexp(nameLower, '(^|[^a-z])o([^a-z]|$)', 'once')), y = 1; return; end
if ~isempty(regexp(nameLower, '(^|[^a-z])c([^a-z]|$)', 'once')), y = 2; return; end
end
