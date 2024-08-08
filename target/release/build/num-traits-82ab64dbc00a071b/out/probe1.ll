; ModuleID = 'probe1.7e09a86c1dbf42b9-cgu.0'
source_filename = "probe1.7e09a86c1dbf42b9-cgu.0"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@alloc_f93507f8ba4b5780b14b2c2584609be0 = private unnamed_addr constant <{ [8 x i8] }> <{ [8 x i8] c"\00\00\00\00\00\00\F0?" }>, align 8
@alloc_ef0a1f828f3393ef691f2705e817091c = private unnamed_addr constant <{ [8 x i8] }> <{ [8 x i8] c"\00\00\00\00\00\00\00@" }>, align 8

; core::f64::<impl f64>::total_cmp
; Function Attrs: inlinehint nonlazybind uwtable
define internal i8 @"_ZN4core3f6421_$LT$impl$u20$f64$GT$9total_cmp17h0cc6252e39ae5620E"(ptr align 8 %self, ptr align 8 %other) unnamed_addr #0 !dbg !17 {
start:
  %other.dbg.spill6 = alloca [8 x i8], align 8
  %self.dbg.spill5 = alloca [8 x i8], align 8
  %self.dbg.spill4 = alloca [8 x i8], align 8
  %self.dbg.spill2 = alloca [8 x i8], align 8
  %other.dbg.spill = alloca [8 x i8], align 8
  %self.dbg.spill = alloca [8 x i8], align 8
  %right = alloca [8 x i8], align 8
  %left = alloca [8 x i8], align 8
  store ptr %self, ptr %self.dbg.spill, align 8
    #dbg_declare(ptr %self.dbg.spill, !26, !DIExpression(), !34)
  store ptr %other, ptr %other.dbg.spill, align 8
    #dbg_declare(ptr %other.dbg.spill, !27, !DIExpression(), !35)
    #dbg_declare(ptr %left, !28, !DIExpression(), !36)
    #dbg_declare(ptr %right, !31, !DIExpression(), !37)
  %self1 = load double, ptr %self, align 8, !dbg !38
  store double %self1, ptr %self.dbg.spill2, align 8, !dbg !38
    #dbg_declare(ptr %self.dbg.spill2, !39, !DIExpression(), !48)
    #dbg_declare(ptr %self.dbg.spill2, !50, !DIExpression(), !57)
  %_4 = bitcast double %self1 to i64, !dbg !59
  store i64 %_4, ptr %left, align 8, !dbg !38
  %self3 = load double, ptr %other, align 8, !dbg !60
  store double %self3, ptr %self.dbg.spill4, align 8, !dbg !60
    #dbg_declare(ptr %self.dbg.spill4, !46, !DIExpression(), !61)
    #dbg_declare(ptr %self.dbg.spill4, !55, !DIExpression(), !63)
  %_7 = bitcast double %self3 to i64, !dbg !65
  store i64 %_7, ptr %right, align 8, !dbg !60
  %_13 = load i64, ptr %left, align 8, !dbg !66
  %_12 = ashr i64 %_13, 63, !dbg !67
  %_10 = lshr i64 %_12, 1, !dbg !68
  %0 = load i64, ptr %left, align 8, !dbg !69
  %1 = xor i64 %0, %_10, !dbg !69
  store i64 %1, ptr %left, align 8, !dbg !69
  %_18 = load i64, ptr %right, align 8, !dbg !70
  %_17 = ashr i64 %_18, 63, !dbg !71
  %_15 = lshr i64 %_17, 1, !dbg !72
  %2 = load i64, ptr %right, align 8, !dbg !73
  %3 = xor i64 %2, %_15, !dbg !73
  store i64 %3, ptr %right, align 8, !dbg !73
  store ptr %left, ptr %self.dbg.spill5, align 8, !dbg !74
    #dbg_declare(ptr %self.dbg.spill5, !75, !DIExpression(), !86)
  store ptr %right, ptr %other.dbg.spill6, align 8, !dbg !88
    #dbg_declare(ptr %other.dbg.spill6, !85, !DIExpression(), !89)
  %_21 = load i64, ptr %left, align 8, !dbg !90
  %_22 = load i64, ptr %right, align 8, !dbg !91
  %4 = icmp sgt i64 %_21, %_22, !dbg !92
  %5 = zext i1 %4 to i8, !dbg !92
  %6 = icmp slt i64 %_21, %_22, !dbg !92
  %7 = zext i1 %6 to i8, !dbg !92
  %_0 = sub nsw i8 %5, %7, !dbg !92
  ret i8 %_0, !dbg !93
}

; probe1::probe
; Function Attrs: nonlazybind uwtable
define void @_ZN6probe15probe17ha805ad019ddc2e7aE() unnamed_addr #1 !dbg !94 {
start:
; call core::f64::<impl f64>::total_cmp
  %_1 = call i8 @"_ZN4core3f6421_$LT$impl$u20$f64$GT$9total_cmp17h0cc6252e39ae5620E"(ptr align 8 @alloc_f93507f8ba4b5780b14b2c2584609be0, ptr align 8 @alloc_ef0a1f828f3393ef691f2705e817091c), !dbg !99
  ret void, !dbg !100
}

attributes #0 = { inlinehint nonlazybind uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #1 = { nonlazybind uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}
!llvm.dbg.cu = !{!5}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 2, !"RtLibUseGOT", i32 1}
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"rustc version 1.82.0-nightly (e57f3090a 2024-08-05)"}
!5 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !6, producer: "clang LLVM (rustc version 1.82.0-nightly (e57f3090a 2024-08-05))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, splitDebugInlining: false, nameTableKind: None)
!6 = !DIFile(filename: "probe1/@/probe1.7e09a86c1dbf42b9-cgu.0", directory: "/home/amondal/.cargo/registry/src/index.crates.io-6f17d22bba15001f/num-traits-0.2.19")
!7 = !{!8}
!8 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Ordering", scope: !10, file: !9, baseType: !12, size: 8, align: 8, flags: DIFlagEnumClass, elements: !13)
!9 = !DIFile(filename: "<unknown>", directory: "")
!10 = !DINamespace(name: "cmp", scope: !11)
!11 = !DINamespace(name: "core", scope: null)
!12 = !DIBasicType(name: "i8", size: 8, encoding: DW_ATE_signed)
!13 = !{!14, !15, !16}
!14 = !DIEnumerator(name: "Less", value: -1)
!15 = !DIEnumerator(name: "Equal", value: 0)
!16 = !DIEnumerator(name: "Greater", value: 1)
!17 = distinct !DISubprogram(name: "total_cmp", linkageName: "_ZN4core3f6421_$LT$impl$u20$f64$GT$9total_cmp17h0cc6252e39ae5620E", scope: !19, file: !18, line: 1465, type: !21, scopeLine: 1465, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, templateParams: !33, retainedNodes: !25)
!18 = !DIFile(filename: "/rustc/e57f3090aec33cdbf66063c866afaa5e1e78b9bb/library/core/src/num/f64.rs", directory: "", checksumkind: CSK_MD5, checksum: "1fdd64dbfd19662d3d36c8fa6d36d0dd")
!19 = !DINamespace(name: "{impl#0}", scope: !20)
!20 = !DINamespace(name: "f64", scope: !11)
!21 = !DISubroutineType(types: !22)
!22 = !{!8, !23, !23}
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&f64", baseType: !24, size: 64, align: 64, dwarfAddressSpace: 0)
!24 = !DIBasicType(name: "f64", size: 64, encoding: DW_ATE_float)
!25 = !{!26, !27, !28, !31}
!26 = !DILocalVariable(name: "self", arg: 1, scope: !17, file: !18, line: 1465, type: !23)
!27 = !DILocalVariable(name: "other", arg: 2, scope: !17, file: !18, line: 1465, type: !23)
!28 = !DILocalVariable(name: "left", scope: !29, file: !18, line: 1466, type: !30, align: 8)
!29 = distinct !DILexicalBlock(scope: !17, file: !18, line: 1466, column: 9)
!30 = !DIBasicType(name: "i64", size: 64, encoding: DW_ATE_signed)
!31 = !DILocalVariable(name: "right", scope: !32, file: !18, line: 1467, type: !30, align: 8)
!32 = distinct !DILexicalBlock(scope: !29, file: !18, line: 1467, column: 9)
!33 = !{}
!34 = !DILocation(line: 1465, column: 22, scope: !17)
!35 = !DILocation(line: 1465, column: 29, scope: !17)
!36 = !DILocation(line: 1466, column: 13, scope: !29)
!37 = !DILocation(line: 1467, column: 13, scope: !32)
!38 = !DILocation(line: 1466, column: 24, scope: !17)
!39 = !DILocalVariable(name: "self", arg: 1, scope: !40, file: !18, line: 1132, type: !24)
!40 = distinct !DILexicalBlock(scope: !41, file: !18, line: 1132, column: 5)
!41 = distinct !DISubprogram(name: "to_bits", linkageName: "_ZN4core3f6421_$LT$impl$u20$f64$GT$7to_bits17ha282a80ad25f929bE", scope: !19, file: !18, line: 1132, type: !42, scopeLine: 1132, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, templateParams: !33, retainedNodes: !45)
!42 = !DISubroutineType(types: !43)
!43 = !{!44, !24}
!44 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
!45 = !{!39, !46}
!46 = !DILocalVariable(name: "self", arg: 1, scope: !47, file: !18, line: 1132, type: !24)
!47 = distinct !DILexicalBlock(scope: !41, file: !18, line: 1132, column: 5)
!48 = !DILocation(line: 1132, column: 26, scope: !40, inlinedAt: !49)
!49 = !DILocation(line: 1466, column: 29, scope: !17)
!50 = !DILocalVariable(name: "rt", arg: 1, scope: !51, file: !18, line: 1154, type: !24)
!51 = distinct !DILexicalBlock(scope: !52, file: !18, line: 1154, column: 9)
!52 = distinct !DISubprogram(name: "rt_f64_to_u64", linkageName: "_ZN4core3f6421_$LT$impl$u20$f64$GT$7to_bits13rt_f64_to_u6417hf953126d60050ae2E", scope: !53, file: !18, line: 1154, type: !42, scopeLine: 1154, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, templateParams: !33, retainedNodes: !54)
!53 = !DINamespace(name: "to_bits", scope: !19)
!54 = !{!50, !55}
!55 = !DILocalVariable(name: "rt", arg: 1, scope: !56, file: !18, line: 1154, type: !24)
!56 = distinct !DILexicalBlock(scope: !52, file: !18, line: 1154, column: 9)
!57 = !DILocation(line: 1154, column: 26, scope: !51, inlinedAt: !58)
!58 = !DILocation(line: 1160, column: 9, scope: !40, inlinedAt: !49)
!59 = !DILocation(line: 1158, column: 22, scope: !51, inlinedAt: !58)
!60 = !DILocation(line: 1467, column: 25, scope: !29)
!61 = !DILocation(line: 1132, column: 26, scope: !47, inlinedAt: !62)
!62 = !DILocation(line: 1467, column: 31, scope: !29)
!63 = !DILocation(line: 1154, column: 26, scope: !56, inlinedAt: !64)
!64 = !DILocation(line: 1160, column: 9, scope: !47, inlinedAt: !62)
!65 = !DILocation(line: 1158, column: 22, scope: !56, inlinedAt: !64)
!66 = !DILocation(line: 1491, column: 20, scope: !32)
!67 = !DILocation(line: 1491, column: 19, scope: !32)
!68 = !DILocation(line: 1491, column: 17, scope: !32)
!69 = !DILocation(line: 1491, column: 9, scope: !32)
!70 = !DILocation(line: 1492, column: 21, scope: !32)
!71 = !DILocation(line: 1492, column: 20, scope: !32)
!72 = !DILocation(line: 1492, column: 18, scope: !32)
!73 = !DILocation(line: 1492, column: 9, scope: !32)
!74 = !DILocation(line: 1494, column: 9, scope: !32)
!75 = !DILocalVariable(name: "self", arg: 1, scope: !76, file: !77, line: 1575, type: !83)
!76 = distinct !DILexicalBlock(scope: !78, file: !77, line: 1575, column: 17)
!77 = !DIFile(filename: "/rustc/e57f3090aec33cdbf66063c866afaa5e1e78b9bb/library/core/src/cmp.rs", directory: "", checksumkind: CSK_MD5, checksum: "ebcf22de2f9b723aa109457819ab8554")
!78 = distinct !DISubprogram(name: "cmp", linkageName: "_ZN4core3cmp5impls48_$LT$impl$u20$core..cmp..Ord$u20$for$u20$i64$GT$3cmp17hc6dbbf44596a6186E", scope: !79, file: !77, line: 1575, type: !81, scopeLine: 1575, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, templateParams: !33, retainedNodes: !84)
!79 = !DINamespace(name: "{impl#79}", scope: !80)
!80 = !DINamespace(name: "impls", scope: !10)
!81 = !DISubroutineType(types: !82)
!82 = !{!8, !83, !83}
!83 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&i64", baseType: !30, size: 64, align: 64, dwarfAddressSpace: 0)
!84 = !{!75, !85}
!85 = !DILocalVariable(name: "other", arg: 2, scope: !76, file: !77, line: 1575, type: !83)
!86 = !DILocation(line: 1575, column: 24, scope: !76, inlinedAt: !87)
!87 = !DILocation(line: 1494, column: 14, scope: !32)
!88 = !DILocation(line: 1494, column: 18, scope: !32)
!89 = !DILocation(line: 1575, column: 31, scope: !76, inlinedAt: !87)
!90 = !DILocation(line: 1576, column: 58, scope: !76, inlinedAt: !87)
!91 = !DILocation(line: 1576, column: 65, scope: !76, inlinedAt: !87)
!92 = !DILocation(line: 1576, column: 21, scope: !76, inlinedAt: !87)
!93 = !DILocation(line: 1495, column: 6, scope: !17)
!94 = distinct !DISubprogram(name: "probe", linkageName: "_ZN6probe15probe17ha805ad019ddc2e7aE", scope: !96, file: !95, line: 1, type: !97, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, templateParams: !33)
!95 = !DIFile(filename: "<anon>", directory: "", checksumkind: CSK_MD5, checksum: "ca821b87a81998bc0a84ab6029e9650c")
!96 = !DINamespace(name: "probe1", scope: null)
!97 = !DISubroutineType(types: !98)
!98 = !{null}
!99 = !DILocation(line: 1, column: 26, scope: !94)
!100 = !DILocation(line: 1, column: 50, scope: !101)
!101 = !DILexicalBlockFile(scope: !94, file: !95, discriminator: 0)
