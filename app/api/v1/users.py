from fastapi import APIRouter, HTTPException, status

from app.models.user import UserCreateRequest, UserLoginRequest, UserResponse, UserUpdateRequest
from app.repository.user_repository import UserNameAlreadyExistsError, UserRepository
from app.utils.passwords import hash_password, is_password_hash, verify_password


users_router = APIRouter()


@users_router.post("/login", response_model=UserResponse)
async def login_user(request: UserLoginRequest):
    user = UserRepository.get_user_by_name(request.user_name)
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    password_matches = (
        verify_password(request.password, user.user_password)
        if is_password_hash(user.user_password)
        else request.password == user.user_password
    )
    if not password_matches:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    return UserResponse(
        user_id=user.user_id,
        user_name=user.user_name,
        user_type=user.user_type,
    )


@users_router.get("/", response_model=list[UserResponse])
async def list_users():
    return UserRepository.list_users()


@users_router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(request: UserCreateRequest):
    try:
        return UserRepository.create_user(
            user_name=request.user_name,
            user_password=hash_password(request.password),
            user_type=request.user_type,
            user_work_describe=request.user_work_describe,
        )
    except UserNameAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail="用户名已存在") from exc


@users_router.put("/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, request: UserUpdateRequest):
    try:
        user = UserRepository.update_user(
            user_id=user_id,
            user_name=request.user_name,
            user_type=request.user_type,
            user_password=hash_password(request.password) if request.password else None,
            user_work_describe=request.user_work_describe,
        )
    except UserNameAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail="用户名已存在") from exc

    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


@users_router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int):
    if not UserRepository.delete_user(user_id):
        raise HTTPException(status_code=404, detail="用户不存在")
