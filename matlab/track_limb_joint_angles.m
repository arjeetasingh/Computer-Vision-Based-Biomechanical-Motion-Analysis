function results = track_limb_joint_angles(videoFile, varargin)
%TRACK_LIMB_JOINT_ANGLES  Track three red markers on a limb and compute joint kinematics.
%
%   results = track_limb_joint_angles('PRI.mp4')
%   results = track_limb_joint_angles('shashank.mp4', 'JointNames', {'shoulder','elbow','wrist'})
%
%   Improvements over the course assignments:
%     * marker identities are kept with a velocity prediction and a distance gate (no per-frame
%       sort by y, so labels cannot jump to another red blob);
%     * short occlusions are bridged by prediction instead of dropping the frame;
%     * tracking glitches are repaired, trajectories are smoothed, angles are signed and have
%       velocities and range of motion;
%     * everything is written to CSV (same column names as the Python tool, so
%       `python -m biomech analyze results/<name>_points.csv --fps <fps>` also works).
%
%   Name-value options
%     'JointNames'  {'hip','knee','ankle'}   proximal, middle, distal marker
%     'OutputDir'   'results'
%     'GatePx'      70      max jump between prediction and detection to accept a match
%     'MaxGap'      15      frames of prediction only before the marker is flagged lost
%     'SmoothSec'   0.25    Savitzky-Golay window in seconds
%     'Facing'      'auto'  'auto' | 'right' | 'left' (direction the limb swings towards)
%     'InitPoints'  []      3x2 [x y] start points, used when frame 1 shows extra red blobs
%     'SaveVideo'   true    write the annotated video
%     'SlowDown'    1       slow-motion factor of the output video
%
%   Requires MATLAB R2019a+ (matchpairs), Image Processing Toolbox, Computer Vision Toolbox
%   (insertShape/insertText).
%   The same logic is implemented and unit-tested in the Python package (../biomech). This MATLAB
%   file was written alongside it but could not be executed in the environment where the repository
%   was prepared, so run it once on your own video and check the first frames before relying on it.

p = inputParser;
addRequired(p, 'videoFile');
addParameter(p, 'JointNames', {'hip', 'knee', 'ankle'});
addParameter(p, 'OutputDir', 'results');
addParameter(p, 'GatePx', 70);
addParameter(p, 'MaxGap', 15);
addParameter(p, 'SmoothSec', 0.25);
addParameter(p, 'Facing', 'auto');
addParameter(p, 'InitPoints', []);
addParameter(p, 'SaveVideo', true);
addParameter(p, 'SlowDown', 1);
parse(p, videoFile, varargin{:});
o = p.Results;
names = o.JointNames;
if ~isfolder(o.OutputDir), mkdir(o.OutputDir); end
[~, baseName] = fileparts(videoFile);

vr = VideoReader(videoFile);
fps = vr.FrameRate;
nEst = max(1, floor(vr.Duration * fps) + 5);
pos = nan(nEst, 3, 2);          % frame x marker x [x y]
status = zeros(nEst, 3);        % 0 detected, 1 predicted, 2 lost
k = 0;

% ---- 1. tracking -------------------------------------------------------------------------
cur = [];  vel = zeros(3, 2);  misses = zeros(3, 1);
while hasFrame(vr)
    frame = readFrame(vr);
    k = k + 1;
    det = detectRed(frame);                              % [x y area] per blob
    if isempty(cur)
        if ~isempty(o.InitPoints)
            cur = o.InitPoints(:, 1:2);
        elseif size(det, 1) >= 3
            [~, idx] = sort(det(:, 3), 'descend');
            keep = det(idx(1:3), 1:2);
            [~, ord] = sort(keep(:, 2));                 % top-to-bottom = proximal-to-distal at frame 1
            cur = keep(ord, :);
        else
            status(k, :) = 2;
            continue;
        end
    else
        predicted = cur + vel;
        matched = false(3, 1);
        newPos = predicted;
        if ~isempty(det)
            C = sqrt((predicted(:, 1) - det(:, 1)').^2 + (predicted(:, 2) - det(:, 2)').^2);   % markers x blobs
            C(C > o.GatePx) = 1e6;
            M = matchpairs(C, 1e5);                      % [marker detection] pairs, minimum total cost
            for r = 1:size(M, 1)
                if C(M(r, 1), M(r, 2)) <= o.GatePx
                    newPos(M(r, 1), :) = det(M(r, 2), 1:2);
                    matched(M(r, 1)) = true;
                end
            end
        end
        vel(matched, :) = 0.7 * vel(matched, :) + 0.3 * (newPos(matched, :) - cur(matched, :));
        misses(matched) = 0;
        misses(~matched) = misses(~matched) + 1;
        status(k, :) = double(~matched)';
        status(k, misses > o.MaxGap) = 2;
        cur = newPos;
    end
    pos(k, :, :) = reshape(cur, 1, 3, 2);
end
n = k;
pos = pos(1:n, :, :);  status = status(1:n, :);
t = ((0:n-1) / fps)';

% ---- 2. clean: repair lost frames and glitches, smooth ----------------------------------------
clean = zeros(n, 3, 2);
glitch = false(n, 3);
win = max(5, 2 * floor(round(o.SmoothSec * fps) / 2) + 1);
for m = 1:3
    for d = 1:2
        y = squeeze(pos(:, m, d));
        y(status(:, m) == 2) = NaN;
        g = isnan(y);
        y2 = fillmissing(y, 'linear', 'EndValues', 'nearest');
        dev = abs(y2 - movmedian(y2, 9));
        if d == 1, devx = dev; else, devy = dev; end
        clean(:, m, d) = y;                                % placeholder, finished below
    end
    bad = hypot(devx, devy) > 25;                          % distance from running median (px)
    glitch(:, m) = bad | isnan(squeeze(pos(:, m, 1))) | status(:, m) == 2;
    for d = 1:2
        y = squeeze(pos(:, m, d));
        y(glitch(:, m)) = NaN;
        y = fillmissing(y, 'pchip', 'EndValues', 'nearest');
        if n >= win
            y = smoothdata(y, 'sgolay', win);
        end
        clean(:, m, d) = y;
    end
end

% ---- 3. joint angles (image coordinates, y down) --------------------------------------------
P = squeeze(clean(:, 1, :));  Mk = squeeze(clean(:, 2, :));  D = squeeze(clean(:, 3, :));
switch lower(o.Facing)
    case 'right', facing = 1;
    case 'left',  facing = -1;
    otherwise
        dx = Mk(:, 1) - P(:, 1);
        if abs(max(dx)) >= abs(min(dx)), facing = 1; else, facing = -1; end
end
segAng = @(a, b) rad2deg(atan2(facing * (b(:, 1) - a(:, 1)), b(:, 2) - a(:, 2)));
proxAngle  = segAng(P, Mk);                                % proximal segment vs vertical, signed
distAngle  = segAng(Mk, D);                                % distal segment vs vertical, signed
v1 = P - Mk;  v2 = D - Mk;
included = rad2deg(acos(max(-1, min(1, sum(v1 .* v2, 2) ./ (vecnorm(v1, 2, 2) .* vecnorm(v2, 2, 2))))));
flexion = 180 - included;
legacyProx = rad2deg(acos(max(-1, min(1, (Mk(:, 2) - P(:, 2)) ./ vecnorm(Mk - P, 2, 2)))));   % assignment definition
legacyDist = abs(rad2deg(atan2(D(:, 2) - Mk(:, 2), D(:, 1) - Mk(:, 1))));                    % assignment definition

dt = 1 / fps;
velOf = @(a) gradient(smoothdata(a, 'sgolay', win), dt);
T = table((0:n-1)', t, proxAngle, flexion, included, distAngle, legacyProx, legacyDist, ...
    velOf(proxAngle), velOf(flexion), velOf(distAngle), ...
    'VariableNames', {'frame', 'time_s', [names{1} '_angle'], [names{2} '_flexion'], [names{2} '_included'], ...
    [names{3} '_segment_angle'], [names{1} '_legacy'], [names{3} '_legacy'], ...
    [names{1} '_angle_vel'], [names{2} '_flexion_vel'], [names{3} '_segment_angle_vel']});
writetable(T, fullfile(o.OutputDir, [baseName '_angles.csv']));

pts = table((0:n-1)', t, 'VariableNames', {'frame', 'time_s'});
for m = 1:3
    pts.([names{m} '_x']) = squeeze(pos(:, m, 1));
    pts.([names{m} '_y']) = squeeze(pos(:, m, 2));
    pts.([names{m} '_status']) = status(:, m);
end
writetable(pts, fullfile(o.OutputDir, [baseName '_points.csv']));

% ---- 4. summary and figures ---------------------------------------------------------------
cols = {proxAngle, flexion, distAngle};
labels = {[names{1} ' segment vs vertical'], [names{2} ' flexion'], [names{3} ' segment vs vertical']};
rangeDeg = zeros(3, 1); minDeg = zeros(3, 1); maxDeg = zeros(3, 1); meanDeg = zeros(3, 1); peakVel = zeros(3, 1);
for i = 1:3
    minDeg(i) = min(cols{i});  maxDeg(i) = max(cols{i});  rangeDeg(i) = maxDeg(i) - minDeg(i);
    meanDeg(i) = mean(cols{i});  peakVel(i) = max(abs(velOf(cols{i})));
end
summary = table(string(labels'), minDeg, maxDeg, rangeDeg, meanDeg, peakVel, ...
    'VariableNames', {'angle', 'min_deg', 'max_deg', 'range_deg', 'mean_deg', 'peak_velocity_dps'});
writetable(summary, fullfile(o.OutputDir, [baseName '_summary.csv']));
disp(summary);
fprintf('Repaired frames per marker: %s | facing = %+d\n', mat2str(sum(glitch)), facing);

fig = figure('Color', 'w', 'Position', [100 100 900 700], 'Visible', 'off');
tiledlayout(3, 1, 'TileSpacing', 'compact');
for i = 1:3
    nexttile; plot(t, cols{i}, 'LineWidth', 1.4); grid on; ylabel([labels{i} ' (deg)']);
end
xlabel('time (s)'); sgtitle(baseName, 'Interpreter', 'none');
saveas(fig, fullfile(o.OutputDir, [baseName '_angles.png'])); close(fig);

% ---- 5. annotated video -----------------------------------------------------------------------
if o.SaveVideo
    vr.CurrentTime = 0;
    vw = VideoWriter(fullfile(o.OutputDir, [baseName '_annotated.mp4']), 'MPEG-4');
    vw.FrameRate = fps / o.SlowDown;  open(vw);
    for f = 1:n
        frame = readFrame(vr);
        pp = [squeeze(clean(f, :, 1))', squeeze(clean(f, :, 2))'];
        frame = insertShape(frame, 'Line', [pp(1, :) pp(2, :); pp(2, :) pp(3, :)], 'LineWidth', 8, 'Color', 'cyan');
        frame = insertShape(frame, 'FilledCircle', [pp, repmat(11, 3, 1)], 'Color', 'yellow', 'Opacity', 1);
        txt = {sprintf('%s %.1f deg', names{1}, proxAngle(f)), ...
               sprintf('%s flex %.1f deg', names{2}, flexion(f)), ...
               sprintf('%s seg %.1f deg', names{3}, distAngle(f))};
        frame = insertText(frame, pp + [16 -10], txt, 'FontSize', 16, 'BoxColor', 'black', ...
            'BoxOpacity', 0.55, 'TextColor', 'white');
        writeVideo(vw, frame);
    end
    close(vw);
end

results = struct('angles', T, 'points', pts, 'summary', summary, 'fps', fps, 'facing', facing);
end

% =================================================================================================
function det = detectRed(frame)
%DETECTRED  Red-sticker blobs: rule from the course code (R > 1.3 G, R > 1.3 B, R > 30).
R = double(frame(:, :, 1)); G = double(frame(:, :, 2)); B = double(frame(:, :, 3));
mask = (R > 1.3 * G) & (R > 1.3 * B) & (R > 30);
mask = imclose(mask, strel('disk', 12));
mask = bwareaopen(mask, 120);
s = regionprops(mask, 'Centroid', 'Area');
if isempty(s)
    det = zeros(0, 3);
else
    det = [cat(1, s.Centroid), cat(1, s.Area)];
end
end
