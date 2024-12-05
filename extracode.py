'''
# Save cropped image of a player
for track_id, player in tracks['players'][0].items():
    bbox = player['bbox']
    frame = video_frames[0]

    # Crop bbox from frame
    cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    # Save the cropped image
    cv2.imwrite(f'cropped_images/cropped_image.jpg', cropped_image)
    break
'''
