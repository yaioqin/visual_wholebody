"""Video timing and camera overlays, independent of Isaac Gym native bindings."""

import numpy as np


def recording_schedule(dt, duration=60.0):
    """Match render_record's one frame per two control steps."""
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Video duration must be a positive number of seconds")
    fps = 1.0 / (2 * dt)
    frames = max(1, round(duration * fps))
    return fps, 2 * frames


def project_world_points(points, view, projection, width, height):
    """Project world points using Isaac Gym's row-vector camera matrices."""
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    clip = homogeneous @ np.asarray(view) @ np.asarray(projection)
    valid = np.isfinite(clip).all(axis=1) & (clip[:, 3] > 1e-6)
    ndc = np.full((len(points), 2), np.nan)
    ndc[valid] = clip[valid, :2] / clip[valid, 3:4]
    valid &= (np.abs(ndc) <= 1).all(axis=1)
    pixels = (ndc * [1, -1] + 1) * [width / 2, height / 2]
    return pixels, valid


def overlay_goal(image, trajectory, current_goal, view, projection):
    """Draw the planned EE path and goals over RGB, including occluded goals.

    Gym's viewer lines are absent from camera sensors. This overlay uses the
    camera's actual matrices and does not add physics actors to the environment.
    All input positions are world coordinates, including for nonzero env origins.
    """
    from PIL import Image, ImageDraw

    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    height, width = image.shape[:2]
    points = np.concatenate((trajectory, np.asarray(current_goal)[None, :]), axis=0)
    pixels, visible = project_world_points(points, view, projection, width, height)
    path_color, goal_color, end_color = (255, 70, 70), (255, 230, 0), (0, 230, 255)
    for j in range(len(trajectory) - 1):
        if visible[j] and visible[j + 1]:
            segment = [tuple(pixels[j]), tuple(pixels[j + 1])]
            draw.line(segment, fill=(20, 20, 20), width=5)
            draw.line(segment, fill=path_color, width=3)
    for index, color, radius in [(-2, end_color, 6), (-1, goal_color, 7)]:
        if visible[index]:
            x, y = pixels[index]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         fill=color, outline=(20, 20, 20), width=2)

    draw.rectangle((8, 8, 176, 61), fill=(20, 20, 20))
    for y, color, label in [(12, goal_color, "Current EE target"),
                            (28, path_color, "Target trajectory"),
                            (44, end_color, "Trajectory endpoint")]:
        draw.line((14, y + 4, 28, y + 4), fill=color, width=3)
        draw.text((34, y), label, fill=(255, 255, 255))
    return np.array(canvas)
