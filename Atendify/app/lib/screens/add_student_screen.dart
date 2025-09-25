import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';
import 'package:fluttertoast/fluttertoast.dart';

class AddStudentScreen extends StatefulWidget {
  const AddStudentScreen({super.key});
  @override
  _AddStudentScreenState createState() => _AddStudentScreenState();
}

class _AddStudentScreenState extends State<AddStudentScreen> {
  final ApiService api = ApiService();
  final TextEditingController _nameController = TextEditingController();
  File? _image;
  bool _loading = false;

  Future<void> _pickImage() async {
    final p = ImagePicker();
    final picked = await p.pickImage(source: ImageSource.camera, imageQuality: 85);
    if (picked != null) setState(() => _image = File(picked.path));
  }

  Future<void> _submit() async {
    final name = _nameController.text.trim();
    if (name.isEmpty || _image == null) {
      Fluttertoast.showToast(msg: "Provide name and photo");
      return;
    }
    setState(() => _loading = true);
    try {
      final ok = await api.addStudent(_image!, name);
      if (ok) {
        Fluttertoast.showToast(msg: "Student added");
        Navigator.pop(context);
      } else {
        Fluttertoast.showToast(msg: "Failed to add student");
      }
    } catch (e) {
      Fluttertoast.showToast(msg: "Error: $e");
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Add Student")),
      body: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          children: [
            TextField(controller: _nameController, decoration: const InputDecoration(labelText: "Student name")),
            const SizedBox(height: 12),
            _image == null ? const Text("No photo selected") : Image.file(_image!, height: 200),
            const SizedBox(height: 12),
            Row(
              children: [
                ElevatedButton.icon(icon: const Icon(Icons.camera_alt), label: const Text("Camera"), onPressed: _pickImage),
                const SizedBox(width: 8),
                ElevatedButton.icon(icon: const Icon(Icons.upload_file), label: const Text("Add"), onPressed: _submit),
              ],
            ),
            if (_loading) const Padding(padding: EdgeInsets.only(top:12), child: CircularProgressIndicator()),
          ],
        ),
      ),
    );
  }
}
