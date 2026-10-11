% BSE 643/443 Applied Biomechanics
% Assignment 2: MATLAB program for marking limb segments and motion
% Name: Arjeeta Singh   Roll: 2502171003 (M. Tech. BMD)
%
% Original course submission (formatting restored, logic unchanged).
% Known limitation: the 3 lowest-y red blobs are re-sorted by y in EVERY frame, so marker labels
% can jump to the wrong blob (e.g. a red sticker on the other leg). See ../track_limb_joint_angles.m.

inputFile = 'PRI.mp4';
videoSource = VideoReader(inputFile);

outputFile = 'PRI_Leg_Fixed_Final.mp4';   % original: a path on the OneDrive desktop
outputVideo = VideoWriter(outputFile, 'MPEG-4');
outputVideo.FrameRate = videoSource.FrameRate;
open(outputVideo);

hFig = figure('Name', 'Leg Tracking: Low-Point Fix');

while hasFrame(videoSource)
    frame = readFrame(videoSource);

    R = double(frame(:,:,1));
    G = double(frame(:,:,2));
    B = double(frame(:,:,3));

    redMask = (R > 1.3 * G) & (R > 1.3 * B) & (R > 30);
    redMask = imclose(redMask, strel('disk', 20));
    redMask = bwareaopen(redMask, 250);
    stats = regionprops(redMask, 'Centroid');
    annotatedFrame = frame;

    if ~isempty(stats)
        allPts = cat(1, stats.Centroid);
        sortedPts = sortrows(allPts, 2);

        if size(sortedPts, 1) > 3
            sortedPts = sortedPts(1:3, :);
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
disp('Video saved successfully.');
