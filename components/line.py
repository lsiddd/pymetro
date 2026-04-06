# ./components/line.py
import pygame
import math
from typing import List, Optional, Tuple, Any
from config import CONFIG
from state import now_ms

class Line:
    def __init__(self, color: str, index: int):
        self.color: Tuple[int, int, int] = self._hex_to_rgb(color)
        self.index: int = index
        self.stations: List[Any] = []
        self.trains: List[Any] = []
        self.active: bool = False
        # --- CHANGE START ---
        self.marked_for_deletion: bool = False
        # --- CHANGE END ---
        self.original_start: Optional[Any] = None
        self.original_end: Optional[Any] = None
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        rgb_values = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (rgb_values[0], rgb_values[1], rgb_values[2])
    
    def add_station(self, station: Any, insert_index: int = -1) -> bool:
        """Add a station to this line"""
        if len(self.stations) == 0:
            self.original_start = station
        
        # --- BUG FIX ---
        # More robust check to prevent adding a station that is already an endpoint,
        # which would create a zero-length segment.
        if len(self.stations) > 0:
            if insert_index == 0 and station == self.stations[0]:
                return False # Trying to add to start, but it's already the start
            is_append = insert_index == -1 or insert_index >= len(self.stations)
            if is_append and station == self.stations[-1]:
                return False # Trying to add to end, but it's already the end
        # --- END FIX ---

        is_closing_loop = (
            len(self.stations) >= 2 and
            station == self.stations[0] and
            (insert_index == -1 or insert_index == len(self.stations))
        )
        
        if station in self.stations and not is_closing_loop:
            return False
        
        if insert_index == 0:
            self.stations.insert(0, station)
            self.original_start = station
        elif 0 < insert_index < len(self.stations):
            self.stations.insert(insert_index, station)
        else:
            self.stations.append(station)
            if not is_closing_loop:
                self.original_end = station
        
        self.active = len(self.stations) >= 2
        self._animate_station_connection(station)
        
        # Mark pathfinding graph as dirty
        from systems.pathfinding import mark_graph_dirty
        mark_graph_dirty()
        
        return True
    
    def remove_end_station(self, station: Any) -> None:
        """Remove a station from the ends of the line"""
        if len(self.stations) < 2:
            return
        
        is_loop = len(self.stations) > 2 and self.stations[0] == self.stations[-1]
        
        try:
            index = self.stations.index(station)
            last_index = len(self.stations) - 1
            
            if is_loop:
                if station == self.original_start and (index == 0 or index == last_index):
                    self.stations.pop()
                    self.original_end = self.stations[-1] if self.stations else None
                elif station == self.original_end:
                    self.stations.pop()
                    self.stations.reverse()
                    self.original_start = self.stations[0] if self.stations else None
                    self.original_end = self.stations[-1] if self.stations else None
            else:
                if index == 0:
                    self.stations.pop(0)
                    self.original_start = self.stations[0] if self.stations else None
                elif index == last_index:
                    self.stations.pop()
                    self.original_end = self.stations[-1] if self.stations else None
        except ValueError:
            pass  # Station not in list
        
        # --- CHANGE START ---
        if len(self.stations) < 2:
            self.marked_for_deletion = True
        else:
            self.active = True
        # --- CHANGE END ---
            
        # Mark pathfinding graph as dirty
        from systems.pathfinding import mark_graph_dirty
        mark_graph_dirty()
    
    def remove_station(self, station: Any) -> None:
        """Remove a station from anywhere in the line"""
        indices = [i for i, s in enumerate(self.stations) if s == station]
        
        if not indices:
            return
        
        is_loop = len(self.stations) > 2 and self.stations[0] == self.stations[-1]
        is_endpoint = not is_loop and (indices[0] == 0 or indices[0] == len(self.stations) - 1)
        
        if is_endpoint:
            self.remove_end_station(station)
            return
        
        if len(indices) > 1:  # Station appears multiple times (loop case)
            self.stations.pop()  # Remove duplicate at end
            self.stations.pop(indices[0])  # Remove from middle
            if self.stations:
                self.stations.append(self.stations[0])  # Re-close loop
        else:
            self.stations.pop(indices[0])
        
        self.active = len(self.stations) >= 2
        # --- CHANGE START ---
        if not self.active:
            self.marked_for_deletion = True
        # --- CHANGE END ---
            
        # Mark pathfinding graph as dirty
        from systems.pathfinding import mark_graph_dirty
        mark_graph_dirty()
    
    def clear_line(self) -> None:
        """Clear the entire line and return resources"""
        from state import game_state
        
        # Calculate refunded bridges
        refunded_bridges = 0
        for i in range(len(self.stations) - 1):
            if self.check_river_crossing(self.stations[i], self.stations[i+1]):
                refunded_bridges += 1
        game_state.bridges += refunded_bridges
        
        # Return trains and carriages to available pool
        for train in self.trains[:]:
            if train in game_state.trains:
                game_state.trains.remove(train)
            game_state.available_trains += 1
            game_state.carriages += train.carriage_count
        
        # Reset line state
        self.stations = []
        self.trains = []
        self.active = False
        # --- CHANGE START ---
        self.marked_for_deletion = False
        # --- CHANGE END ---
        self.original_start = None
        self.original_end = None
        
        # Mark pathfinding graph as dirty
        from systems.pathfinding import mark_graph_dirty
        mark_graph_dirty()
    
    def check_river_crossing(self, station1: Any, station2: Any) -> bool:
        """Check if line segment crosses a river"""
        from state import game_state
        
        if not station1 or not station2:
            return False
        
        for river in game_state.rivers:
            if self._line_intersects_polygon(
                station1.x, station1.y,
                station2.x, station2.y,
                river.points
            ):
                return True
        return False
    
    def _line_intersects_polygon(self, x1: float, y1: float, x2: float, y2: float, polygon_points: List[Any]) -> bool:
        """Check if line intersects with polygon"""
        try:
            for i in range(len(polygon_points)):
                p1 = polygon_points[i]
                p2 = polygon_points[(i + 1) % len(polygon_points)]
                
                # Handle both dict and object formats
                p1x = p1['x'] if isinstance(p1, dict) else p1.x
                p1y = p1['y'] if isinstance(p1, dict) else p1.y
                p2x = p2['x'] if isinstance(p2, dict) else p2.x
                p2y = p2['y'] if isinstance(p2, dict) else p2.y
                
                if self._line_intersects_line(x1, y1, x2, y2, p1x, p1y, p2x, p2y):
                    return True
            return False
        except (KeyError, AttributeError, TypeError) as e:
            print(f"Line: Error checking polygon intersection: {e}")
            return False
    
    def _line_intersects_line(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float) -> bool:
        """Check if two line segments intersect"""
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if denom == 0:
            return False
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
        
        return 0 < t < 1 and 0 < u < 1
    
    def _animate_station_connection(self, station: Any) -> None:
        """Animate station connection effect"""
        if station:
            station.connection_animation = {
                'start_time': now_ms(),
                'duration': 300
            }
    
    def find_insertion_point(self, station: Any) -> int:
        """Find best insertion point for a station in the line"""
        if len(self.stations) < 2:
            return -1
        
        min_distance = float('inf')
        insert_index = -1
        
        for i in range(len(self.stations) - 1):
            s1 = self.stations[i]
            s2 = self.stations[i + 1]
            distance = self._distance_to_line_segment(station, s1, s2)
            
            if distance < min_distance and distance < 50:
                min_distance = distance
                insert_index = i + 1
        
        return insert_index
    
    def _distance_to_line_segment(self, point: Any, line_start: Any, line_end: Any) -> float:
        """Calculate distance from point to line segment"""
        A = point.x - line_start.x
        B = point.y - line_start.y
        C = line_end.x - line_start.x
        D = line_end.y - line_start.y
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        param = -1
        if len_sq != 0:
            param = dot / len_sq
        
        if param < 0:
            xx, yy = line_start.x, line_start.y
        elif param > 1:
            xx, yy = line_end.x, line_end.y
        else:
            xx = line_start.x + param * C
            yy = line_start.y + param * D
        
        dx = point.x - xx
        dy = point.y - yy
        return math.sqrt(dx * dx + dy * dy)
    
    # Radius used for the Bézier corner fillet (px)
    _FILLET_R: float = 14.0

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the line using 45°/90°-routed segments with smooth Bézier corners.

        When multiple lines share the same station-pair segment they are drawn as
        true parallel paths: the base waypoints are computed once from the raw
        station coordinates, then each line's copy is offset via _offset_path()
        which moves the elbow to the geometric intersection of the two offset
        sub-segments, guaranteeing constant spacing along every sub-segment.
        """
        if (len(self.stations) < 2 and not self.marked_for_deletion) or not self.active:
            return

        from state import game_state

        for i in range(len(self.stations) - 1):
            s1 = self.stations[i]
            s2 = self.stations[i + 1]

            # Collect lines that contain s1↔s2 as consecutive stations, sorted for
            # a stable rank so every frame the same line gets the same offset.
            shared = sorted(
                [l for l in game_state.lines if l.active and l._has_segment(s1, s2)],
                key=lambda l: l.index,
            )
            n = len(shared)
            my_rank = next((j for j, l in enumerate(shared) if l.index == self.index), 0)
            # Centre the bundle around the station axis
            offset = (my_rank - (n - 1) / 2.0) * (CONFIG.LINE_WIDTH + 3)

            # Compute the canonical (zero-offset) waypoints from station positions
            base = self._compute_metro_waypoints((s1.x, s1.y), (s2.x, s2.y))

            # Offset every point in the path, preserving true parallelism
            pts = self._offset_path(base, offset) if abs(offset) > 0.01 else list(base)

            line_color = self.color
            if self.marked_for_deletion:
                line_color = (
                    int(self.color[0] * 0.5 + 128 * 0.5),
                    int(self.color[1] * 0.5 + 128 * 0.5),
                    int(self.color[2] * 0.5 + 128 * 0.5)
                )

            crosses_river = self.check_river_crossing(s1, s2)
            dashed = crosses_river or self.marked_for_deletion
            w = int(CONFIG.LINE_WIDTH * 0.8) if crosses_river else CONFIG.LINE_WIDTH

            self._draw_metro_path(screen, pts, line_color, w, dashed)
    
    # ------------------------------------------------------------------
    # Metro-style 45°/90° routing helpers
    # ------------------------------------------------------------------

    def _compute_metro_waypoints(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        """Return 2 or 3 waypoints that route p1→p2 at 45°/90° angles.

        If the segment is already axis-aligned or perfectly diagonal, return
        the two endpoints directly.  Otherwise insert one elbow so that the
        path travels diagonally first, then axis-aligned.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        adx, ady = abs(dx), abs(dy)

        # Straight or near-straight: skip elbow
        if adx < 2 or ady < 2 or abs(adx - ady) < 2:
            return [p1, p2]

        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        diag = min(adx, ady)
        elbow = (p1[0] + diag * sx, p1[1] + diag * sy)
        return [p1, elbow, p2]

    def _draw_metro_path(
        self,
        screen: pygame.Surface,
        pts: List[Tuple[float, float]],
        color: Tuple[int, int, int],
        width: int,
        dashed: bool = False,
    ) -> None:
        """Draw a list of waypoints as a metro-style path.

        For a 3-point path, the corner at the middle point is smoothed with a
        quadratic Bézier curve (fillet).
        """
        if len(pts) == 2:
            if dashed:
                self._draw_dashed_line(screen, pts[0][0], pts[0][1],
                                       pts[1][0], pts[1][1], color, width)
            else:
                pygame.draw.line(screen, color,
                                 (int(pts[0][0]), int(pts[0][1])),
                                 (int(pts[1][0]), int(pts[1][1])), width)
            return

        # 3-point path: draw two segments meeting at an elbow with a fillet
        p0, p1, p2 = pts
        r = self._FILLET_R

        d01 = self._unit_vec(p1[0] - p0[0], p1[1] - p0[1])
        d12 = self._unit_vec(p2[0] - p1[0], p2[1] - p1[1])

        # Tangent points where the fillet meets the straight segments
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
        color: Tuple[int, int, int],
        pa: Tuple[float, float],
        pb: Tuple[float, float],
        pc: Tuple[float, float],
        width: int,
        dashed: bool = False,
        n: int = 10,
    ) -> None:
        """Quadratic Bézier from pa through pb to pc (used for smooth corners)."""
        pts = []
        for i in range(n + 1):
            t = i / n
            x = (1 - t) ** 2 * pa[0] + 2 * (1 - t) * t * pb[0] + t ** 2 * pc[0]
            y = (1 - t) ** 2 * pa[1] + 2 * (1 - t) * t * pb[1] + t ** 2 * pc[1]
            pts.append((int(x), int(y)))
        for i in range(len(pts) - 1):
            pygame.draw.line(screen, color, pts[i], pts[i + 1], width)

    @staticmethod
    def _unit_vec(dx: float, dy: float) -> Tuple[float, float]:
        d = math.sqrt(dx * dx + dy * dy)
        if d < 1e-6:
            return (0.0, 0.0)
        return (dx / d, dy / d)

    def get_train_waypoints(self, s1: Any, s2: Any) -> List[Tuple[float, float]]:
        """Return the actual polyline the train should follow between s1 and s2.

        Computes the same offset as draw() and expands the Bézier corner into
        sampled points so trains follow the visual path exactly.
        """
        from state import game_state

        shared = sorted(
            [l for l in game_state.lines if l.active and l._has_segment(s1, s2)],
            key=lambda l: l.index,
        )
        n = len(shared)
        my_rank = next((j for j, l in enumerate(shared) if l.index == self.index), 0)
        offset = (my_rank - (n - 1) / 2.0) * (CONFIG.LINE_WIDTH + 3)

        base = self._compute_metro_waypoints((s1.x, s1.y), (s2.x, s2.y))
        pts = self._offset_path(base, offset) if abs(offset) > 0.01 else list(base)

        if len(pts) == 2:
            return pts

        # 3-point path: expand the Bézier corner into polyline samples
        p0, p1, p2 = pts
        r = self._FILLET_R
        d01 = self._unit_vec(p1[0] - p0[0], p1[1] - p0[1])
        d12 = self._unit_vec(p2[0] - p1[0], p2[1] - p1[1])
        ta = (p1[0] - d01[0] * r, p1[1] - d01[1] * r)
        tb = (p1[0] + d12[0] * r, p1[1] + d12[1] * r)

        bezier_pts: List[Tuple[float, float]] = []
        n_b = 10
        for i in range(n_b + 1):
            t = i / n_b
            x = (1 - t) ** 2 * ta[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * tb[0]
            y = (1 - t) ** 2 * ta[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * tb[1]
            bezier_pts.append((x, y))

        return [p0] + bezier_pts + [p2]

    def _has_segment(self, s1: Any, s2: Any) -> bool:
        """Return True if s1 and s2 appear as consecutive stations on this line."""
        stations = self.stations
        for i in range(len(stations) - 1):
            if (stations[i] is s1 and stations[i + 1] is s2) or \
               (stations[i] is s2 and stations[i + 1] is s1):
                return True
        return False

    def _offset_path(
        self,
        pts: List[Tuple[float, float]],
        offset: float,
    ) -> List[Tuple[float, float]]:
        """Offset a 2- or 3-point metro path by *offset* pixels, maintaining
        true geometric parallelism.

        For a 2-point (straight) path both endpoints are shifted by the same
        perpendicular vector → trivially parallel.

        For a 3-point path (diagonal + axis-aligned), the elbow of the offset
        path is the intersection of:
          • the line through (p0_offset) in direction d01
          • the line through (p2_offset) in direction -d12
        This guarantees that both sub-segments of the offset path are exactly
        *offset* pixels away from the corresponding sub-segments of the original.
        """
        if len(pts) == 2:
            p0, p2 = pts
            n = self._perp_unit(p2[0] - p0[0], p2[1] - p0[1])
            return [
                (p0[0] + offset * n[0], p0[1] + offset * n[1]),
                (p2[0] + offset * n[0], p2[1] + offset * n[1]),
            ]

        p0, p1, p2 = pts  # p1 is the elbow

        d01 = self._unit_vec(p1[0] - p0[0], p1[1] - p0[1])
        d12 = self._unit_vec(p2[0] - p1[0], p2[1] - p1[1])

        # CCW perpendicular of each sub-segment
        n01 = (-d01[1], d01[0])
        n12 = (-d12[1], d12[0])

        # Offset the two endpoints
        p0_off = (p0[0] + offset * n01[0], p0[1] + offset * n01[1])
        p2_off = (p2[0] + offset * n12[0], p2[1] + offset * n12[1])

        # The offset elbow is the intersection of:
        #   line through p0_off in direction d01
        #   line through p2_off in direction d12 (approaching from p2 side)
        # We parameterise both lines from p1 shifted to the two normals:
        a = (p1[0] + offset * n01[0], p1[1] + offset * n01[1])
        b = (p1[0] + offset * n12[0], p1[1] + offset * n12[1])

        p1_off = self._line_intersect_2d(
            a[0], a[1], d01[0], d01[1],
            b[0], b[1], d12[0], d12[1],
        )
        if p1_off is None:
            # Parallel directions (degenerate corner) – average as fallback
            p1_off = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

        return [p0_off, p1_off, p2_off]

    @staticmethod
    def _perp_unit(dx: float, dy: float) -> Tuple[float, float]:
        """CCW perpendicular unit vector."""
        d = math.sqrt(dx * dx + dy * dy)
        if d < 1e-6:
            return (0.0, -1.0)
        return (-dy / d, dx / d)

    @staticmethod
    def _line_intersect_2d(
        px1: float, py1: float, dx1: float, dy1: float,
        px2: float, py2: float, dx2: float, dy2: float,
    ) -> Optional[Tuple[float, float]]:
        """Intersection of two infinite lines given as point + direction.

        Returns None when lines are parallel (denom ≈ 0).
        """
        denom = dx1 * dy2 - dy1 * dx2
        if abs(denom) < 1e-9:
            return None
        t = ((px2 - px1) * dy2 - (py2 - py1) * dx2) / denom
        return (px1 + t * dx1, py1 + t * dy1)

    def _draw_dashed_line(self, screen: pygame.Surface, x1: float, y1: float, x2: float, y2: float, color: Tuple[int, int, int], width: int) -> None:
        """Draw a dashed line"""
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
            
            start_x = x1 + dx * start_ratio
            start_y = y1 + dy * start_ratio
            end_x = x1 + dx * end_ratio
            end_y = y1 + dy * end_ratio
            
            pygame.draw.line(screen, color, (start_x, start_y), (end_x, end_y), width)