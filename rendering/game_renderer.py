# ./rendering/game_renderer.py
"""
All pygame rendering logic for the game world.

Receives simulation objects (Station, Line, Train, Passenger, River) and
reads their state to produce pixels. No simulation state is modified here.
"""

import pygame
import math
from config import CONFIG, STATION_TYPES
from state import now_ms

# Radius used for the Bézier corner fillet (px) — must match Line._FILLET_R
_FILLET_R: float = 14.0


class GameRenderer:
    def render(self, screen: pygame.Surface, game_state) -> pygame.Surface | None:
        """Draw the game world into an intermediate surface and return it.

        The caller composites the surface onto the screen (with zoom + centering)
        so that overlays drawn afterwards land on the world surface before the
        final scale, keeping them correctly aligned with game objects at all
        zoom levels.
        """
        sw, sh = screen.get_width(), screen.get_height()
        world_surf = pygame.Surface((sw, sh))
        world_surf.fill((244, 241, 233))

        for river in game_state.rivers:
            self.draw_river(world_surf, river)
        for line in game_state.lines:
            self.draw_line(world_surf, line, game_state.lines)
        for station in game_state.stations:
            self.draw_station(world_surf, station)
        for train in game_state.trains:
            self.draw_train(world_surf, train)

        return world_surf

    # ------------------------------------------------------------------
    # River
    # ------------------------------------------------------------------

    def draw_river(self, screen: pygame.Surface, river) -> None:
        if len(river.points) < 3:
            return
        pygame_points = [(p['x'], p['y']) for p in river.points]
        pygame.draw.polygon(screen, (173, 216, 230), pygame_points)
        pygame.draw.polygon(screen, (135, 206, 250), pygame_points, 2)

    # ------------------------------------------------------------------
    # Line
    # ------------------------------------------------------------------

    def draw_line(self, screen: pygame.Surface, line, all_lines) -> None:
        """Draw a line using 45°/90°-routed segments with smooth Bézier corners.

        When multiple lines share the same station-pair segment they are drawn
        as true parallel paths offset by a stable rank.
        """
        if (len(line.stations) < 2 and not line.marked_for_deletion) or not line.active:
            return

        for i in range(len(line.stations) - 1):
            s1 = line.stations[i]
            s2 = line.stations[i + 1]

            shared = sorted(
                [l for l in all_lines if l.active and l._has_segment(s1, s2)],
                key=lambda l: l.index,
            )
            n = len(shared)
            my_rank = next((j for j, l in enumerate(shared) if l.index == line.index), 0)
            offset = (my_rank - (n - 1) / 2.0) * (CONFIG.LINE_WIDTH + 3)

            base = line._compute_metro_waypoints((s1.x, s1.y), (s2.x, s2.y))
            pts = line._offset_path(base, offset) if abs(offset) > 0.01 else list(base)

            line_color = line.color
            if line.marked_for_deletion:
                line_color = (
                    int(line.color[0] * 0.5 + 128 * 0.5),
                    int(line.color[1] * 0.5 + 128 * 0.5),
                    int(line.color[2] * 0.5 + 128 * 0.5),
                )

            crosses_river = line.check_river_crossing(s1, s2)
            dashed = crosses_river or line.marked_for_deletion
            w = int(CONFIG.LINE_WIDTH * 0.8) if crosses_river else CONFIG.LINE_WIDTH

            self._draw_metro_path(screen, pts, line_color, w, dashed)

    def _draw_metro_path(
        self,
        screen: pygame.Surface,
        pts,
        color,
        width: int,
        dashed: bool = False,
    ) -> None:
        if len(pts) == 2:
            if dashed:
                self._draw_dashed_line(screen, pts[0][0], pts[0][1],
                                       pts[1][0], pts[1][1], color, width)
            else:
                pygame.draw.line(screen, color,
                                 (int(pts[0][0]), int(pts[0][1])),
                                 (int(pts[1][0]), int(pts[1][1])), width)
            return

        p0, p1, p2 = pts
        r = _FILLET_R
        d01 = _unit_vec(p1[0] - p0[0], p1[1] - p0[1])
        d12 = _unit_vec(p2[0] - p1[0], p2[1] - p1[1])
        ta = (p1[0] - d01[0] * r, p1[1] - d01[1] * r)
        tb = (p1[0] + d12[0] * r, p1[1] + d12[1] * r)

        if dashed:
            self._draw_dashed_line(screen, p0[0], p0[1], ta[0], ta[1], color, width)
            self._draw_bezier_corner(screen, color, ta, p1, tb, width, dashed=True)
            self._draw_dashed_line(screen, tb[0], tb[1], p2[0], p2[1], color, width)
        else:
            pygame.draw.line(screen, color,
                             (int(p0[0]), int(p0[1])), (int(ta[0]), int(ta[1])), width)
            self._draw_bezier_corner(screen, color, ta, p1, tb, width)
            pygame.draw.line(screen, color,
                             (int(tb[0]), int(tb[1])), (int(p2[0]), int(p2[1])), width)

    def _draw_bezier_corner(
        self,
        screen: pygame.Surface,
        color,
        pa,
        pb,
        pc,
        width: int,
        dashed: bool = False,
        n: int = 10,
    ) -> None:
        pts = []
        for i in range(n + 1):
            t = i / n
            x = (1 - t) ** 2 * pa[0] + 2 * (1 - t) * t * pb[0] + t ** 2 * pc[0]
            y = (1 - t) ** 2 * pa[1] + 2 * (1 - t) * t * pb[1] + t ** 2 * pc[1]
            pts.append((int(x), int(y)))
        for i in range(len(pts) - 1):
            pygame.draw.line(screen, color, pts[i], pts[i + 1], width)

    def _draw_dashed_line(
        self,
        screen: pygame.Surface,
        x1: float, y1: float, x2: float, y2: float,
        color,
        width: int,
    ) -> None:
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx * dx + dy * dy)
        if distance == 0:
            return
        dash_length = 8
        gap_length = 4
        total_length = dash_length + gap_length
        num_dashes = int(distance / total_length)
        for i in range(num_dashes + 1):
            start_ratio = (i * total_length) / distance
            end_ratio = min((i * total_length + dash_length) / distance, 1.0)
            if start_ratio >= 1.0:
                break
            sx = x1 + dx * start_ratio
            sy = y1 + dy * start_ratio
            ex = x1 + dx * end_ratio
            ey = y1 + dy * end_ratio
            pygame.draw.line(screen, color, (sx, sy), (ex, ey), width)

    # ------------------------------------------------------------------
    # Station
    # ------------------------------------------------------------------

    def draw_station(self, screen: pygame.Surface, station) -> None:
        if station.overcrowd_start_time:
            elapsed = now_ms() - station.overcrowd_start_time
            progress = elapsed / CONFIG.OVERCROWD_TIME
            pulse = int(CONFIG.STATION_RADIUS + 8 + math.sin(now_ms() / 150) * 5)
            glow_size = pulse * 2 + 4
            glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            cx = glow_size // 2
            pygame.draw.circle(glow_surface, (211, 47, 47, 50), (cx, cx), pulse)
            screen.blit(glow_surface, (int(station.x) - cx, int(station.y) - cx))
            if progress < 1:
                angle = progress * 2 * math.pi
                pygame.draw.arc(
                    screen, (211, 47, 47),
                    (station.x - CONFIG.STATION_RADIUS - 8,
                     station.y - CONFIG.STATION_RADIUS - 8,
                     2 * (CONFIG.STATION_RADIUS + 8),
                     2 * (CONFIG.STATION_RADIUS + 8)),
                    -math.pi / 2, -math.pi / 2 + angle, 4,
                )

        self._draw_station_shape(screen, station)

        if station.is_interchange:
            pygame.draw.circle(screen, (51, 51, 51),
                               (station.x, station.y), CONFIG.STATION_RADIUS + 8, 6)
            if station.animate_upgrade:
                elapsed = now_ms() - station.animate_upgrade['start_time']
                progress = elapsed / station.animate_upgrade['duration']
                if progress < 1:
                    pulse = 1 + 0.3 * math.sin(progress * math.pi * 4)
                    radius = int((CONFIG.STATION_RADIUS + 15) * pulse)
                    pygame.draw.circle(screen, (76, 175, 80),
                                       (station.x, station.y), radius, 4)
                else:
                    station.animate_upgrade = None

        if station.delivery_animation:
            elapsed = now_ms() - station.delivery_animation['start_time']
            progress = elapsed / station.delivery_animation['duration']
            if progress < 1:
                scale = 1 + 0.5 * (1 - progress)
                alpha = int(255 * (1 - progress))
                radius = int(CONFIG.STATION_RADIUS * scale)
                glow_surface = pygame.Surface(
                    (radius * 2 + 10, radius * 2 + 10), pygame.SRCALPHA
                )
                pygame.draw.circle(glow_surface, (76, 175, 80, alpha),
                                   (radius + 5, radius + 5), radius, 3)
                screen.blit(glow_surface,
                            (station.x - radius - 5, station.y - radius - 5))
            else:
                station.delivery_animation = None

        self._draw_passengers(screen, station)

    def _draw_station_shape(self, screen: pygame.Surface, station) -> None:
        fill_color = (244, 241, 233)
        border_color = (51, 51, 51)
        radius = CONFIG.STATION_RADIUS
        x, y = station.x, station.y

        if station.type == STATION_TYPES.CIRCLE:
            pygame.draw.circle(screen, fill_color, (x, y), radius)
            pygame.draw.circle(screen, border_color, (x, y), radius, 4)

        elif station.type == STATION_TYPES.TRIANGLE:
            h = radius * 1.7
            points = [
                (x, y - h * 0.6),
                (x - h * 0.5, y + h * 0.4),
                (x + h * 0.5, y + h * 0.4),
            ]
            pygame.draw.polygon(screen, fill_color, points)
            pygame.draw.polygon(screen, border_color, points, 4)

        elif station.type == STATION_TYPES.SQUARE:
            size = radius * 1.6
            rect = pygame.Rect(x - size / 2, y - size / 2, size, size)
            pygame.draw.rect(screen, fill_color, rect)
            pygame.draw.rect(screen, border_color, rect, 4)

        elif station.type == STATION_TYPES.PENTAGON:
            self._draw_polygon(screen, x, y, 5, radius * 1.4, fill_color, border_color)

        elif station.type == STATION_TYPES.DIAMOND:
            size = radius * 1.4
            points = [
                (x, y - size),
                (x + size, y),
                (x, y + size),
                (x - size, y),
            ]
            pygame.draw.polygon(screen, fill_color, points)
            pygame.draw.polygon(screen, border_color, points, 4)

        elif station.type == STATION_TYPES.STAR:
            self._draw_star(screen, x, y, radius * 1.5, fill_color, border_color)

        elif station.type == STATION_TYPES.CROSS:
            self._draw_cross(screen, x, y, radius * 1.4, fill_color, border_color)

    def _draw_polygon(self, screen, x, y, sides, radius, fill_color, border_color):
        points = []
        for i in range(sides):
            angle = (i * 2 * math.pi) / sides - math.pi / 2
            points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
        pygame.draw.polygon(screen, fill_color, points)
        pygame.draw.polygon(screen, border_color, points, 4)

    def _draw_star(self, screen, x, y, radius, fill_color, border_color):
        outer_radius = radius
        inner_radius = radius * 0.4
        spikes = 5
        points = []
        for i in range(spikes * 2):
            angle = (i * math.pi) / spikes - math.pi / 2
            r = outer_radius if i % 2 == 0 else inner_radius
            points.append((x + r * math.cos(angle), y + r * math.sin(angle)))
        pygame.draw.polygon(screen, fill_color, points)
        pygame.draw.polygon(screen, border_color, points, 4)

    def _draw_cross(self, screen, x, y, size, fill_color, border_color):
        thickness = size * 0.3
        h_rect = pygame.Rect(x - size / 2, y - thickness / 2, size, thickness)
        pygame.draw.rect(screen, fill_color, h_rect)
        pygame.draw.rect(screen, border_color, h_rect, 4)
        v_rect = pygame.Rect(x - thickness / 2, y - size / 2, thickness, size)
        pygame.draw.rect(screen, fill_color, v_rect)
        pygame.draw.rect(screen, border_color, v_rect, 4)

    def _draw_passengers(self, screen: pygame.Surface, station) -> None:
        passenger_area = CONFIG.STATION_RADIUS * 0.9
        cols = 3
        slot_size = CONFIG.PASSENGER_SIZE + 2
        start_x = station.x - (cols / 2 - 0.5) * slot_size
        start_y = station.y - passenger_area / 2 + CONFIG.PASSENGER_SIZE / 2
        for i, passenger in enumerate(station.passengers):
            if i < station.capacity:
                row = i // cols
                col = i % cols
                px = start_x + col * slot_size
                py = start_y + row * slot_size
                self.draw_passenger_at(screen, passenger, px, py)
            elif i == station.capacity:
                self.draw_passenger_at(screen, passenger,
                                       station.x, station.y + CONFIG.STATION_RADIUS + 15)

    # ------------------------------------------------------------------
    # Passenger
    # ------------------------------------------------------------------

    def draw_passenger_at(self, screen: pygame.Surface, passenger, x: float, y: float) -> None:
        color = (51, 51, 51)
        size = CONFIG.PASSENGER_SIZE
        dest = passenger.destination

        if dest == STATION_TYPES.CIRCLE:
            pygame.draw.circle(screen, color, (int(x), int(y)), size // 2)

        elif dest == STATION_TYPES.TRIANGLE:
            points = [
                (x, y - size // 2),
                (x - size // 2, y + size // 2),
                (x + size // 2, y + size // 2),
            ]
            pygame.draw.polygon(screen, color, points)

        elif dest == STATION_TYPES.SQUARE:
            rect = pygame.Rect(x - size // 2, y - size // 2, size, size)
            pygame.draw.rect(screen, color, rect)

        elif dest == STATION_TYPES.PENTAGON:
            self._draw_passenger_polygon(screen, x, y, 5, size // 2, color)

        elif dest == STATION_TYPES.DIAMOND:
            points = [
                (x, y - size // 2),
                (x + size // 2, y),
                (x, y + size // 2),
                (x - size // 2, y),
            ]
            pygame.draw.polygon(screen, color, points)

        elif dest == STATION_TYPES.STAR:
            self._draw_passenger_star(screen, x, y, size // 2, color)

        elif dest == STATION_TYPES.CROSS:
            thickness = size * 0.3
            h_rect = pygame.Rect(x - size // 2, y - thickness // 2, size, thickness)
            pygame.draw.rect(screen, color, h_rect)
            v_rect = pygame.Rect(x - thickness // 2, y - size // 2, thickness, size)
            pygame.draw.rect(screen, color, v_rect)

    def _draw_passenger_polygon(self, screen, x, y, sides, radius, color):
        points = []
        for i in range(sides):
            angle = (i * 2 * math.pi) / sides - math.pi / 2
            points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
        pygame.draw.polygon(screen, color, points)

    def _draw_passenger_star(self, screen, x, y, radius, color):
        outer_radius = radius
        inner_radius = radius * 0.4
        spikes = 5
        points = []
        for i in range(spikes * 2):
            angle = (i * math.pi) / spikes - math.pi / 2
            r = outer_radius if i % 2 == 0 else inner_radius
            points.append((x + r * math.cos(angle), y + r * math.sin(angle)))
        pygame.draw.polygon(screen, color, points)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def draw_train(self, screen: pygame.Surface, train) -> None:
        if not train.line.active or len(train.line.stations) < 2:
            return

        width, height = CONFIG.TRAIN_WIDTH, CONFIG.TRAIN_HEIGHT
        stations = train.line.stations

        current_st = stations[train.current_station_index] if train.current_station_index < len(stations) else None
        next_st = stations[train.next_station_index] if train.next_station_index < len(stations) else None

        if not current_st:
            return
        next_st = next_st or current_st

        if train._path_pts and len(train._path_pts) >= 2:
            angle = train._get_angle_on_path(train.progress)
        elif next_st and next_st != current_st:
            angle = math.atan2(next_st.y - current_st.y, next_st.x - current_st.x)
        else:
            angle = 0.0

        n_units = 1 + train.carriage_count
        gap = 5
        total_width = n_units * width + (n_units - 1) * gap
        train_surface = pygame.Surface((total_width, height), pygame.SRCALPHA)

        for u in range(n_units):
            ux = u * (width + gap)
            pygame.draw.rect(train_surface, train.line.color, (ux, 0, width, height))
            pygame.draw.rect(train_surface, (51, 51, 51), (ux, 0, width, height), 2)
            if u > 0:
                pygame.draw.line(train_surface, (51, 51, 51),
                                 (ux - gap, height // 2), (ux, height // 2), 1)

        if train.passengers:
            max_dots = min(len(train.passengers), 4)
            for i in range(max_dots):
                dot_x = width // 4 + (i % 2) * width // 4
                dot_y = height // 4 + (i // 2) * height // 4
                pygame.draw.circle(train_surface, (255, 255, 255), (dot_x, dot_y), 2)

        rotated_surface = pygame.transform.rotate(train_surface, -math.degrees(angle))
        rotated_rect = rotated_surface.get_rect(center=(int(train.x), int(train.y)))
        screen.blit(rotated_surface, rotated_rect)


# ------------------------------------------------------------------
# Module-level geometry helpers (used internally by renderer)
# ------------------------------------------------------------------

def _unit_vec(dx: float, dy: float):
    d = math.sqrt(dx * dx + dy * dy)
    if d < 1e-6:
        return (0.0, 0.0)
    return (dx / d, dy / d)
