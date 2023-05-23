# -*- coding: utf-8 -*-
import sqlite3
import numpy as np
import cv2

class ImageMatcher:

    def __init__(self, db_path):
        self.db_path = db_path
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()
        self.conn = sqlite3.connect(self.db_path)

    def __del__(self):
        self.conn.close()

    def blob_to_array(self, blob, dtype, shape=(-1,)):
        return np.frombuffer(blob, dtype=dtype).reshape(shape)

    def get_data_from_db(self, image_name):
        with self.conn as conn:
            cur = conn.cursor()
            query = """
                SELECT k.data, d.data
                FROM keypoints k
                JOIN descriptors d on k.image_id = d.image_id
                WHERE k.image_id IN (
                    SELECT DISTINCT image_id
                    FROM images
                    WHERE name = ?            
                )
                """
            cur.execute(query, (image_name,))
            return cur.fetchone()

    def process_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        return image, keypoints, descriptors

    def match_images(self, descriptors1, keypoints2, descriptors2):
        descriptors1 = descriptors1.astype(np.float32)
        matches = self.bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])
        return good_matches

    def run(self, image_name, new_image_path):
        keypoints_blob, descriptors_blob = self.get_data_from_db(image_name)

        keypoints1 = self.blob_to_array(keypoints_blob, np.float32, (-1, 6))
        descriptors1 = self.blob_to_array(descriptors_blob, np.uint8, (-1, 128))

        new_image, new_keypoints, new_descriptors = self.process_image(new_image_path)

        good_matches = self.match_images(descriptors1, new_keypoints, new_descriptors)

        keypoints_colmap = [cv2.KeyPoint(x[0], x[1], x[2], x[3]) for x in keypoints1]

        matched_image = cv2.drawMatchesKnn(new_image, keypoints_colmap, new_image, new_keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow('Matched Images', matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    db_path = db_path
    image_name = image_path
    new_image_path = new_image_path

    matcher = ImageMatcher(db_path)
    matcher.run(image_name, new_image_path)
