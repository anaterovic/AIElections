
Dim sName
Dim fso
Dim fol

' kreira filesystem
Set fso = WScript.CreateObject("Scripting.FileSystemObject")
    
Set fol = fso.GetFolder(".")


For Each fil In fol.Files
    ' provjera za space
    If InStr(1, fil.Name, " ") <> 0 Then
        ' replace spejsa s underskorom
        sName = Replace(fil.Name, " ", "_")
        ' rename datoteke
        fil.Name = sName
    End If
Next
'è
For Each fil In fol.Files
    If InStr(1, fil.Name, "è") <> 0 Then
        sName = Replace(fil.Name, "è", "c")
        fil.Name = sName
    End If
   
Next
'æ
For Each fil In fol.Files
    If InStr(1, fil.Name, "æ") <> 0 Then
        sName = Replace(fil.Name, "æ", "c")
        fil.Name = sName
    End If
Next    
'È
For Each fil In fol.Files
    If InStr(1, fil.Name, "È") <> 0 Then
        sName = Replace(fil.Name, "È", "C")
        fil.Name = sName
    End If
Next    
'Æ
For Each fil In fol.Files
    If InStr(1, fil.Name, "Æ") <> 0 Then
        sName = Replace(fil.Name, "Æ", "C")
        fil.Name = sName
    End If
Next     
'Š
For Each fil In fol.Files
    If InStr(1, fil.Name, "Š") <> 0 Then
        sName = Replace(fil.Name, "Š", "S")
        fil.Name = sName
    End If
Next   
'š
For Each fil In fol.Files
    If InStr(1, fil.Name, "š") <> 0 Then
        sName = Replace(fil.Name, "š", "s")
        fil.Name = sName
    End If
Next
'Ð
For Each fil In fol.Files
    If InStr(1, fil.Name, "Ð") <> 0 Then
        sName = Replace(fil.Name, "Ð", "D")
        fil.Name = sName
    End If
Next
'ð
For Each fil In fol.Files
    If InStr(1, fil.Name, "ð") <> 0 Then
        sName = Replace(fil.Name, "ð", "d")
        fil.Name = sName
    End If
Next
'Ž
For Each fil In fol.Files
    If InStr(1, fil.Name, "Ž") <> 0 Then
        sName = Replace(fil.Name, "Ž", "Z")
        fil.Name = sName
    End If
Next
'ž
For Each fil In fol.Files
    If InStr(1, fil.Name, "ž") <> 0 Then
        sName = Replace(fil.Name, "ž", "z")
        fil.Name = sName
    End If
Next

WScript.Echo "Gotovo!"