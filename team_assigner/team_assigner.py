from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {} # {player_id: team_id}

    def get_clustering_model(self, image):
        # Reshape image to 2D array
        image_2d = image.reshape(-1, 3)

        # Perform KMeans clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1).fit(image_2d) # n_init=1 to speed up the process
        
        return kmeans

    # Same steps as in color_analysis done on ipynb
    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:image.shape[0]//2, :]

        # Get clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 if non_player_cluster == 0 else 0   # Same as 1 - non_player_cluster

        # Get the player color
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for track_id, player_detections in player_detections.items():
            bbox = player_detections['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1).fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1  # Player is 0 or 1, Team is 1 or 2

        self.player_team_dict[player_id] = team_id

        return team_id