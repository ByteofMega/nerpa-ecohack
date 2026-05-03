from utils import GeoUtils


class MatchingSystem:
    @staticmethod
    def compare_detections(sat_detected, uw_detected, drone_lat, drone_lon, target_lat, target_lon):
        if target_lat is None or target_lon is None:
            return "ОШИБКА КООРДИНАТ", "red"

        dist_diff = GeoUtils.calculate_distance(drone_lat, drone_lon, target_lat, target_lon)

        if sat_detected and uw_detected and dist_diff < 50:
            return "МУСОР ПОДТВЕРЖДЕН", "#10b981"  # success color
        elif sat_detected and not uw_detected:
            return "ЛОЖНАЯ ТРЕВОГА", "#ef4444"  # danger color
        else:
            return "НЕОПРЕДЕЛЕННОСТЬ", "#f59e0b"  # warning color
