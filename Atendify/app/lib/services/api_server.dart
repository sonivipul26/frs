  import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'models.dart';

class ApiService {
  // CHANGE THIS to your backend address:
  // - For Android emulator use 10.0.2.2
  // - For iOS simulator use localhost
  // - For phone on same Wi-Fi use your PC's LAN IP (e.g. http://192.168.1.42:8000)
  final String baseUrl;

  ApiService({this.baseUrl = "http://10.0.2.2:8000"});

  Future<AttendanceResponse> processAttendance(File imageFile,
      {String teacherName = "", String subject = "", String className = ""}) async {
    var uri = Uri.parse("$baseUrl/api/process-attendance");
    var request = http.MultipartRequest("POST", uri);
    request.fields['teacher_name'] = teacherName;
    request.fields['subject'] = subject;
    request.fields['class_name'] = className;
    request.files.add(await http.MultipartFile.fromPath('photo', imageFile.path));

    final streamed = await request.send();
    final resp = await http.Response.fromStream(streamed);

    if (resp.statusCode == 200) {
      final data = json.decode(resp.body);
      return AttendanceResponse.fromJson(data);
    } else {
      throw Exception("Server error: ${resp.statusCode} ${resp.body}");
    }
  }

  Future<bool> addStudent(File imageFile, String studentName,
      {String rollNumber = "", String className = ""}) async {
    var uri = Uri.parse("$baseUrl/api/add-student");
    var request = http.MultipartRequest("POST", uri);
    request.fields['student_name'] = studentName;
    request.fields['roll_number'] = rollNumber;
    request.fields['class_name'] = className;
    request.files.add(await http.MultipartFile.fromPath('student_photo', imageFile.path));

    final streamed = await request.send();
    final resp = await http.Response.fromStream(streamed);
    return resp.statusCode == 200;
  }

  Future<List<String>> getStudents() async {
    final uri = Uri.parse("$baseUrl/api/students");
    final resp = await http.get(uri);
    if (resp.statusCode == 200) {
      final data = json.decode(resp.body);
      final list = (data['students'] as List).map((e) => e['name'] as String).toList();
      return list;
    } else {
      throw Exception("Failed to fetch students");
    }
  }

  Future<Map<String, dynamic>> getSession(String sessionId) async {
    final uri = Uri.parse("$baseUrl/api/session/$sessionId");
    final resp = await http.get(uri);
    if (resp.statusCode == 200) {
      return json.decode(resp.body) as Map<String, dynamic>;
    } else {
      throw Exception("Failed to fetch session");
    }
  }
}
