import argparse
import glob
import os
import time

import cv2
import gymnasium as gym
import hockey.hockey_env as h_env
import numpy as np

from framework.registry import discover_agents, load_opponents


class VideoOverlayWrapper(gym.Wrapper):
    def __init__(self, env, a1_name, a2_name):
        super().__init__(env)
        self.a1_name = a1_name
        self.a2_name = a2_name
        self.score = [0, 0, 0]

    @property
    def render_mode(self):
        return self.env.render_mode

    def render(self):
        try:
            frame = self.env.render(mode="rgb_array")
        except TypeError:
            frame = self.env.render()

        if frame is None or not isinstance(frame, np.ndarray):
            return frame

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w, _ = frame.shape

        header_height = 60
        header = np.zeros((header_height, w, 3), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2

        color_orange = (0, 100, 255)
        color_blue = (255, 170, 50)
        color_white = (255, 255, 255)

        y_pos = 40

        cv2.putText(
            header, f"{self.a1_name}", (20, y_pos), font, scale, color_orange, thickness
        )

        text_size_right = cv2.getTextSize(f"{self.a2_name}", font, scale, thickness)[0]
        cv2.putText(
            header,
            f"{self.a2_name}",
            (w - text_size_right[0] - 20, y_pos),
            font,
            scale,
            color_blue,
            thickness,
        )

        score_text = f"{self.score[0]} - {self.score[1]} - {self.score[2]}"
        text_size_score = cv2.getTextSize(score_text, font, scale, thickness)[0]
        cv2.putText(
            header,
            score_text,
            ((w - text_size_score[0]) // 2, y_pos),
            font,
            scale,
            color_white,
            thickness,
        )

        combined_frame = np.vstack((header, frame))

        return cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)


def load_agent(agent_name: str):
    if agent_name.lower() in ["weak", "strong"]:

        class BasicOpponentWrapper:
            def __init__(self, is_weak):
                self.agent = h_env.BasicOpponent(weak=is_weak)

            def act(self, obs, **kwargs):
                action = self.agent.act(obs[0])
                return np.expand_dims(action, axis=0)

            def reset(self):
                pass

        return BasicOpponentWrapper(is_weak=(agent_name.lower() == "weak"))

    class ArgumentConfig:
        def __init__(self, agent_name: str):
            self.name = "competition"
            self.agent = agent_name
            self.mode = "NORMAL"
            self.total_steps = 0
            self.seed = 0
            self.opponent = []
            self.kwargs = {}
            self.parallel_envs = 1
            self.backup_frequency = 0
            self.eval_games = 0

    config = ArgumentConfig(agent_name)
    agents = load_opponents([config.agent], config)
    return agents[0]


def run_simulation(args):
    env = h_env.HockeyEnv()
    env.render_mode = "rgb_array"
    overlay_env = VideoOverlayWrapper(env, args.a1_name, args.a2_name)

    video_folder = ""
    if args.record:
        video_folder = f"recordings/{args.a1_name}_vs_{args.a2_name}"
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            overlay_env,
            video_folder=video_folder,
            episode_trigger=lambda x: True,
            name_prefix="match",
        )
    else:
        env = overlay_env

    discover_agents()
    player_1 = load_agent(args.agent1)
    player_2 = load_agent(args.agent2)

    wins, losses, draws = 0, 0, 0
    window_name = f"{args.a1_name} vs {args.a2_name}"
    window_initialized = False

    for i in range(args.rounds):
        obs, info = env.reset()
        done = False
        player_1.reset()
        player_2.reset()

        while not done:
            obs_2 = env.unwrapped.obs_agent_two()

            obs_batch = np.expand_dims(obs, axis=0)
            obs_2_batch = np.expand_dims(obs_2, axis=0)

            a1 = np.squeeze(player_1.act(obs_batch, inference=True))
            a2 = np.squeeze(player_2.act(obs_2_batch, inference=True))

            obs, reward, terminated, truncated, info = env.step(np.hstack([a1, a2]))
            done = terminated or truncated

            if not args.headless:
                frame = overlay_env.render()
                if frame is not None:
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    if not window_initialized:
                        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                        try:
                            import tkinter as tk

                            root = tk.Tk()
                            root.withdraw()
                            screen_width = root.winfo_screenwidth()
                            screen_height = root.winfo_screenheight()
                            root.destroy()
                            frame_h, frame_w, _ = bgr_frame.shape
                            x = (screen_width - frame_w) // 2
                            y = (screen_height - frame_h) // 2

                            cv2.moveWindow(window_name, x, y)
                        except ImportError:
                            pass

                        window_initialized = True

                    cv2.imshow(window_name, bgr_frame)
                    cv2.waitKey(15)

        winner = info.get("winner", 0)

        if winner == 1:
            wins += 1
            overlay_env.score[0] += 1
        elif winner == -1:
            losses += 1
            overlay_env.score[2] += 1
        else:
            draws += 1
            overlay_env.score[1] += 1

        if not args.headless:
            time.sleep(0.5)

    print(f"\n--- Final: {args.a1_name} {wins} : {draws} : {losses} {args.a2_name} ---")

    if not args.headless:
        cv2.destroyAllWindows()

    env.close()

    if args.record and video_folder:
        stitch_videos(video_folder)


def stitch_videos(video_folder):
    print("\nStitching games into a single video...")

    try:
        from moviepy import VideoFileClip, concatenate_videoclips
    except ImportError:
        from moviepy.editor import VideoFileClip, concatenate_videoclips

    video_files = glob.glob(os.path.join(video_folder, "match-episode-*.mp4"))

    if not video_files:
        print("No videos found to stitch.")
        return

    video_files = sorted(
        video_files, key=lambda x: int(x.split("-episode-")[-1].split(".mp4")[0])
    )

    try:
        clips = [VideoFileClip(vf) for vf in video_files]

        final_clip = concatenate_videoclips(clips)

        output_path = os.path.join(video_folder, "final_stitched_match.mp4")

        final_clip.write_videofile(output_path, codec="libx264")

        for clip in clips:
            clip.close()
        final_clip.close()

        print(f"Success! Final stitched video saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred while stitching videos: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent1", "-a1", type=str, required=True)
    parser.add_argument("--agent2", "-a2", type=str, required=True)
    parser.add_argument("--a1-name", "-a1n", type=str, default="Left Agent")
    parser.add_argument("--a2-name", "-a2n", type=str, default="Right Agent")
    parser.add_argument("--rounds", "-r", type=int, default=1)
    parser.add_argument(
        "--record", "-v", action="store_true", help="Save video of games"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Deactivate live window rendering"
    )

    args = parser.parse_args()
    run_simulation(args)
