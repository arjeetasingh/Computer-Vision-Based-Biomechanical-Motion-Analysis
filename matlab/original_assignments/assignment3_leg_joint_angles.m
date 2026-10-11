% BSE 643/443 Applied Biomechanics
% Assignment 3: MATLAB program extension for joint angles (leg: hip, knee, ankle)
% Name: Arjeeta Singh   Roll: 2502171003 (M. Tech. BMD)
%
% Original course submission (formatting restored; the unfinished last line "fprint;" is now a
% proper fprintf, and the output path is relative).

inputFile = 'PRI.mp4';
videoSource = VideoReader(inputFile);

outputFile = 'PRI_Triple_Angle_Tracked.mp4';   % original: a path on the OneDrive desktop
outputVideo = VideoWriter(outputFile, 'MPEG-4');
outputVideo.FrameRate = videoSource.FrameRate / 2;   % 2x slow motion for analysis
open(outputVideo);

hFig = figure('Name', 'Leg Analysis: Hip, Knee & Ankle');

while hasFrame(videoSource)
    frame = readFrame(videoSource);

    R = double(frame(:,:,1)); G = double(frame(:,:,2)); B = double(frame(:,:,3));
    redMask = (R > 1.3 * G) & (R > 1.3 * B) & (R > 30);
    redMask = imclose(redMask, strel('disk', 20));
    redMask = bwareaopen(redMask, 250);
    stats = regionprops(redMask, 'Centroid');

    annotatedFrame = frame;

    if ~isempty(stats)
        allPts = cat(1, stats.Centroid);
        sortedPts = sortrows(allPts, 2);   % vertical sort: hip -> knee -> ankle

        if size(sortedPts, 1) > 3
            sortedPts = sortedPts(1:3, :);
        end

        if size(sortedPts, 1) == 3
            H = sortedPts(1,:);   % hip
            K = sortedPts(2,:);   % knee
            A = sortedPts(3,:);   % ankle

            vH = K - H;
            refV = [0, 100];
            angleH = rad2deg(acos(dot(vH, refV) / (norm(vH) * norm(refV))));

            vK1 = H - K;
            vK2 = A - K;
            angleK = rad2deg(acos(dot(vK1, vK2) / (norm(vK1) * norm(vK2))));

            vA = A - K;
            angleA = rad2deg(atan2(vA(2), vA(1)));

            tK = linspace(atan2d(vK1(2), vK1(1)), atan2d(vK2(2), vK2(1)), 20);
            for k = 1:length(tK)-1
                annotatedFrame = insertShape(annotatedFrame, 'Line', ...
                    [K(1)+50*cosd(tK(k)) K(2)+50*sind(tK(k)) K(1)+50*cosd(tK(k+1)) K(2)+50*sind(tK(k+1))], ...
                    'LineWidth', 5, 'Color', 'cyan');
            end

            annotatedFrame = insertText(annotatedFrame, H + [40, -20], sprintf('H: %.1f^o', angleH), ...
                'FontSize', 18, 'BoxColor', 'red', 'TextColor', 'white');
            annotatedFrame = insertText(annotatedFrame, K + [40, 20], sprintf('K: %.1f^o', angleK), ...
                'FontSize', 18, 'BoxColor', 'blue', 'TextColor', 'white');
            annotatedFrame = insertText(annotatedFrame, A + [40, -20], sprintf('A: %.1f^o', abs(angleA)), ...
                'FontSize', 18, 'BoxColor', 'green', 'TextColor', 'white');
        end

        annotatedFrame = insertShape(annotatedFrame, 'FilledCircle', ...
            [sortedPts, repmat(14, size(sortedPts,1), 1)], 'Color', 'yellow', 'Opacity', 1);

        if size(sortedPts, 1) >= 2
            for i = 1 : size(sortedPts, 1) - 1
                linePts = [sortedPts(i,1) sortedPts(i,2) sortedPts(i+1,1) sortedPts(i+1,2)];
                annotatedFrame = insertShape(annotatedFrame, 'Line', linePts, ...
                    'LineWidth', 12, 'Color', 'cyan');
            end
        end

        labels = {'Hip', 'Knee', 'Ankle'};
        for j = 1:min(size(sortedPts, 1), 3)
            annotatedFrame = insertText(annotatedFrame, sortedPts(j, :) + [35, 0], labels{j}, ...
                'FontSize', 22, 'BoxColor', 'yellow', 'BoxOpacity', 0.6, 'TextColor', 'black');
        end
    end

    imshow(annotatedFrame);
    drawnow;
    writeVideo(outputVideo, annotatedFrame);
    if ~ishandle(hFig), break; end
end

close(outputVideo);
fprintf('Saved: %s\n', outputFile);
