% BSE 643/443 Applied Biomechanics
% Assignment 3: MATLAB program extension for joint angles (arm: shoulder, elbow, wrist)
% Name: Shashank Shinde   Roll: 2502171020 (M. Tech. BMD)
%
% Original course submission (formatting restored, logic unchanged; output path made relative).
% Markers are found in the first frame only and then followed with a KLT point tracker, so the
% identities cannot swap. The weakness is the opposite one: a lost point stops the analysis for
% that frame (all(isFound) is false) and the tracker never re-detects.

inputFile = 'shashank.mp4';
videoSource = VideoReader(inputFile);

outputFile = 'Shashank_Triple_Angle_Tracked.mp4';   % original: a path on the OneDrive desktop
outputVideo = VideoWriter(outputFile, 'MPEG-4');
outputVideo.FrameRate = videoSource.FrameRate / 2;   % 2x slow motion for clarity
open(outputVideo);

firstFrame = readFrame(videoSource);
redIntensity = imsubtract(firstFrame(:,:,1), rgb2gray(firstFrame));
binaryMask = redIntensity > 50;
binaryMask = bwareaopen(imopen(binaryMask, strel('disk', 5)), 100);

stats = regionprops(binaryMask, 'Centroid');

if isempty(stats)
    error('First frame markers not detected! Ensure stickers are visible at 0:00.');
end

% Initial points: [Shoulder; Elbow; Wrist]
initialPoints = sortrows(cat(1, stats.Centroid), 2);

tracker = vision.PointTracker('MaxBidirectionalError', 2);
initialize(tracker, initialPoints, firstFrame);

hFig = figure('Name', 'Arm Analysis: Shoulder, Elbow & Wrist');

while hasFrame(videoSource)
    frame = readFrame(videoSource);
    [points, isFound] = step(tracker, frame);
    annotatedFrame = frame;

    if all(isFound)
        S = points(1,:);   % shoulder
        E = points(2,:);   % elbow
        W = points(3,:);   % wrist

        % 1. Shoulder angle (relative to vertical)
        vS = E - S;
        refV = [0, 100];
        angleS = rad2deg(acos(dot(vS, refV) / (norm(vS) * norm(refV))));

        % 2. Elbow angle (internal joint angle)
        vE1 = S - E;
        vE2 = W - E;
        angleE = rad2deg(acos(dot(vE1, vE2) / (norm(vE1) * norm(vE2))));

        % 3. Wrist angle (orientation)
        vW = W - E;
        angleW = rad2deg(atan2(vW(2), vW(1)));

        % Arc sign at the elbow
        tE = linspace(atan2d(vE1(2), vE1(1)), atan2d(vE2(2), vE2(1)), 20);
        for k = 1:length(tE)-1
            annotatedFrame = insertShape(annotatedFrame, 'Line', ...
                [E(1)+45*cosd(tE(k)) E(2)+45*sind(tE(k)) E(1)+45*cosd(tE(k+1)) E(2)+45*sind(tE(k+1))], ...
                'LineWidth', 4, 'Color', 'cyan');
        end

        % Display angles
        annotatedFrame = insertText(annotatedFrame, S + [40, -20], sprintf('S: %.1f^o', angleS), ...
            'FontSize', 18, 'BoxColor', 'red', 'TextColor', 'white');
        annotatedFrame = insertText(annotatedFrame, E + [40, 20], sprintf('E: %.1f^o', angleE), ...
            'FontSize', 18, 'BoxColor', 'blue', 'TextColor', 'white');
        annotatedFrame = insertText(annotatedFrame, W + [40, -20], sprintf('W: %.1f^o', abs(angleW)), ...
            'FontSize', 18, 'BoxColor', 'green', 'TextColor', 'white');

        % Draw skeleton
        annotatedFrame = insertShape(annotatedFrame, 'FilledCircle', ...
            [points, repmat(20, size(points,1), 1)], 'Color', 'yellow');
        linePts = [S(1) S(2) E(1) E(2); E(1) E(2) W(1) W(2)];
        annotatedFrame = insertShape(annotatedFrame, 'Line', linePts, 'LineWidth', 12, 'Color', 'cyan');

        annotatedFrame = insertText(annotatedFrame, points, {'Shoulder', 'Elbow', 'Wrist'}, ...
            'FontSize', 18, 'BoxOpacity', 0.4);
    end

    imshow(annotatedFrame);
    drawnow;
    writeVideo(outputVideo, annotatedFrame);
    if ~ishandle(hFig), break; end
end

close(outputVideo);
release(tracker);
fprintf('Saved: %s\n', outputFile);
