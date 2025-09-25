import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'package:fluttertoast/fluttertoast.dart';

class StudentListScreen extends StatefulWidget {
  const StudentListScreen({super.key});
  @override
  _StudentListScreenState createState() => _StudentListScreenState();
}

class _StudentListScreenState extends State<StudentListScreen> {
  final ApiService api = ApiService();
  List<String> _students = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadStudents();
  }

  Future<void> _loadStudents() async {
    setState(() => _loading = true);
    try {
      final list = await api.getStudents();
      setState(() => _students = list);
    } catch (e) {
      Fluttertoast.showToast(msg: "Failed to load students: $e");
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(title: const Text("Students")),
        body: _loading
            ? const Center(child: CircularProgressIndicator())
            : RefreshIndicator(
                onRefresh: _loadStudents,
                child: ListView.builder(
                    itemCount: _students.length,
                    itemBuilder: (_, i) => ListTile(
                          leading: const Icon(Icons.person),
                          title: Text(_students[i]),
                        ))));
  }
}
