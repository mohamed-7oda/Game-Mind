import numpy as np
import cv2
import os
from utils.video_utils import read_video

from tracking.trackers import Tracker
from tracking.team_assigner import TeamAssigner
from tracking.player_ball_assigner import PlayerBallAssigner
from tracking.camera_movement_estimator import CameraMovementEstimator
from tracking.view_transformer import ViewTransformer
from tracking.speed_and_distance_estimator import SpeedAndDistance_Estimator


def save_video_mp4(frames, output_path, fps=25):
    height, width, _ = frames[0].shape

    temp_avi = output_path.replace(".mp4", "_temp.avi")

    # Step 1: Save as AVI (stable)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_avi, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

    # Step 2: Convert to REAL MP4 (H.264)
    final_path = output_path

    os.system(
        f'ffmpeg -y -i "{temp_avi}" -vcodec libx264 -pix_fmt yuv420p "{final_path}"'
    )

    os.remove(temp_avi)

    return final_path


def run_tracking(video_path, output_path):
    try:
        print("Reading video...")
        video_frames = read_video(video_path)

        if len(video_frames) == 0:
            raise ValueError("Video has no frames!")

        print("Initializing tracker...")
        tracker = Tracker("models/best.pt")

        print("Getting tracks...")
        tracks = tracker.get_object_tracks(video_frames, read_from_stub=False)
        tracker.add_position_to_tracks(tracks)

        print("Camera movement...")
        camera_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement = camera_estimator.get_camera_movement(video_frames)

        print("View transform...")
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)

        print("Interpolating ball...")
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        print("Speed & distance...")
        speed_estimator = SpeedAndDistance_Estimator()
        speed_estimator.add_speed_and_distance_to_tracks(tracks)

        print("Team assignment...")
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

        for frame_num, player_track in enumerate(tracks["players"]):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(
                    video_frames[frame_num],
                    track["bbox"],
                    player_id
                )
                track["team"] = team
                track["team_color"] = team_assigner.team_colors.get(team, (0, 255, 0))

        print("Ball assignment...")
        player_assigner = PlayerBallAssigner()
        team_ball_control = []

        for frame_num, player_track in enumerate(tracks["players"]):
            ball_dict = tracks["ball"][frame_num]

            if 1 in ball_dict:
                ball_bbox = ball_dict[1]["bbox"]
                assigned = player_assigner.assign_ball_to_player(player_track, ball_bbox)
            else:
                assigned = -1

            if assigned != -1:
                tracks["players"][frame_num][assigned]["has_ball"] = True
                team_ball_control.append(tracks["players"][frame_num][assigned]["team"])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

        team_ball_control = np.array(team_ball_control)

        print("Drawing...")
        output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
        output_frames = camera_estimator.draw_camera_movement(output_frames, camera_movement)
        output_frames = speed_estimator.draw_speed_and_distance(output_frames, tracks)

        print("Saving video (MP4)...")

        # ✅ FORCE MP4 output
        output_path = output_path.replace(".avi", ".mp4")
        save_video_mp4(output_frames, output_path)

        # 🔥 MEMORY CLEANUP
        del video_frames
        del output_frames

        print("✅ DONE!")

        return output_path  # ✅ useful for backend

    except Exception as e:
        print("❌ ERROR in tracking pipeline:", str(e))
        raise e


# ==============================
# TEST
# ==============================
if __name__ == "__main__":
    run_tracking("test.mp4", "output.mp4")