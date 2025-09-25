class AttendanceResponse {
  final bool success;
  final String message;
  final List<String> present;
  final List<String> absent;
  final String attendanceRate;
  final String annotatedImage; // base64 string (may be empty)
  final String sessionId;

  AttendanceResponse({
    required this.success,
    required this.message,
    required this.present,
    required this.absent,
    required this.attendanceRate,
    required this.annotatedImage,
    required this.sessionId,
  });

  factory AttendanceResponse.fromJson(Map<String, dynamic> json) {
    return AttendanceResponse(
      success: json['success'] ?? false,
      message: json['message'] ?? '',
      present: List<String>.from(json['present'] ?? []),
      absent: List<String>.from(json['absent'] ?? []),
      attendanceRate: json['attendance_rate'] ?? '',
      annotatedImage: json['annotated_image'] ?? '',
      sessionId: json['session_id'] ?? '',
    );
  }
}
