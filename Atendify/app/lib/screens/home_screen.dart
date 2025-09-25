import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';
import 'attendance_result_screen.dart';
import 'add_student_screen.dart';
import 'student_list_screen.dart';
import 'package:fluttertoast/fluttertoast.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ApiService api = ApiService();
  bool _loading = false;

  Future<void> _captureAndSend() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.camera, imageQuality: 85);
    if (picked == null) return;
    final file = File(picked.path);

    setState(() => _loading = true);
    try {
      final result = await api.processAttendance(file);
      Navigator.push(context, MaterialPageRoute(builder: (_) => AttendanceResultScreen(attendance: result)));
    } catch (e) {
      Fluttertoast.showToast(msg: "Failed: $e");
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> _pickFromGallery() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery, imageQuality: 85);
    if (picked == null) return;
    final file = File(picked.path);

    setState(() => _loading = true);
    try {
      final result = await api.processAttendance(file);
      Navigator.push(context, MaterialPageRoute(builder: (_) => AttendanceResultScreen(attendance: result)));
    } catch (e) {
      Fluttertoast.showToast(msg: "Failed: $e");
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Attendance Home"),
        actions: [
          IconButton(
            icon: const Icon(Icons.people),
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const StudentListScreen())),
            tooltip: 'Students'
          ),
          IconButton(
            icon: const Icon(Icons.person_add),
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const AddStudentScreen())),
            tooltip: 'Add student'
          ),
        ],
      ),
      body: Center(
        child: _loading
            ? const CircularProgressIndicator()
            : Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  ElevatedButton.icon(
                      icon: const Icon(Icons.camera_alt),
                      label: const Text("Take Classroom Photo"),
                      onPressed: _captureAndSend),
                  const SizedBox(height: 12),
                  ElevatedButton.icon(
                      icon: const Icon(Icons.photo_library),
                      label: const Text("Pick Photo from Gallery"),
                      onPressed: _pickFromGallery),
                ],
              ),
      ),
    );
  }
}
